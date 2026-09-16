#!/usr/bin/env python3
"""Audit visual consistency of reconstructed longitudinal NIfTI volumes.

This is an image-derived QC screen for the reconstructed dataset. It compares
each active reconstructed exam against the patient's MAMA-MIA anchor exam when
that anchor is active, otherwise against the earliest active reconstructed exam.

The script is intentionally conservative: low similarity is a "review this"
signal, not proof that a time point was reconstructed from the wrong series.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

try:
    from skimage.metrics import structural_similarity
except Exception:  # pragma: no cover - optional dependency fallback
    structural_similarity = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECON_AUDIT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "reconstruction_consistency_audit"
    / "reconstructed_exam_audit.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "volume_visual_consistency"
)


@dataclass
class VolumeFeatures:
    path: str
    phase_selection_status: str
    load_error: str
    shape: tuple[int, int, int] | None
    canonical_shape: tuple[int, int, int] | None
    spacing: tuple[float, float, float] | None
    canonical_spacing: tuple[float, float, float] | None
    fov: tuple[float, float, float] | None
    canonical_fov: tuple[float, float, float] | None
    axcodes: str
    nifti_header_view: str
    phase_count: int
    nonzero_fraction: float | None
    intensity_p1: float | None
    intensity_p50: float | None
    intensity_p99: float | None
    bbox_center_fraction: tuple[float, float, float] | None
    bbox_extent_fraction: tuple[float, float, float] | None
    normalized_volume: np.ndarray | None
    projection_vector: np.ndarray | None
    projection_images: tuple[np.ndarray, np.ndarray, np.ndarray] | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare reconstructed time-point volumes against each patient's "
            "anchor/earliest reconstructed exam using NIfTI geometry and "
            "image-derived visual similarity metrics."
        )
    )
    parser.add_argument("--reconstruction-audit-csv", type=Path, default=DEFAULT_RECON_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--phase-index",
        type=int,
        default=0,
        help="Reconstructed phase index to compare. Phase 0 is safest for anatomy QC.",
    )
    parser.add_argument(
        "--target-shape",
        default="64,64,32",
        help="Downsampled 3D comparison shape as X,Y,Z.",
    )
    parser.add_argument(
        "--projection-size",
        type=int,
        default=96,
        help="Size of each 2D MIP projection used for projection correlation/SSIM.",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Optional smoke-test limit on the number of patients.",
    )
    parser.add_argument(
        "--fov-rel-threshold",
        type=float,
        default=0.35,
        help="Flag if max relative physical field-of-view difference exceeds this.",
    )
    parser.add_argument(
        "--spacing-rel-threshold",
        type=float,
        default=0.75,
        help="Flag if max relative voxel-spacing difference exceeds this.",
    )
    parser.add_argument(
        "--visual-score-threshold",
        type=float,
        default=0.25,
        help="Flag non-reference exams below this combined visual consistency score.",
    )
    parser.add_argument(
        "--mip-corr-threshold",
        type=float,
        default=0.20,
        help="Flag low visual similarity only when projection correlation is below this.",
    )
    parser.add_argument(
        "--volume-corr-threshold",
        type=float,
        default=0.05,
        help="Flag low visual similarity only when 3D volume correlation is below this.",
    )
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_target_shape(text: str) -> tuple[int, int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError("--target-shape must have three comma-separated integers")
    shape = tuple(int(part) for part in parts)
    if any(value <= 0 for value in shape):
        raise ValueError("--target-shape values must be positive")
    return shape  # type: ignore[return-value]


def safe_float(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return float("nan")
    return float(value)


def tuple_to_text(values: tuple[int, int, int] | tuple[float, float, float] | None) -> str:
    if values is None:
        return ""
    formatted = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            formatted.append(str(int(value)))
        else:
            formatted.append(f"{float(value):.6g}")
    return "x".join(formatted)


def compact_counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value if value not in (None, "") else "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def parse_exam_datetime(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(value, format="%m-%d-%Y", errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.max
    return parsed


def phase_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"phase(\d+)", path.name)
    if match:
        return int(match.group(1)), path.name
    return 10_000, path.name


def list_phase_paths(exam_dir: Path) -> list[Path]:
    if not exam_dir.is_dir():
        return []
    return sorted(exam_dir.glob("volume_phase*.nii.gz"), key=phase_sort_key)


def choose_phase_path(exam_dir_text: object, phase_index: int) -> tuple[Path | None, int, str]:
    if exam_dir_text is None or pd.isna(exam_dir_text):
        return None, 0, "missing_exam_dir"
    exam_dir = Path(str(exam_dir_text))
    if not exam_dir.is_dir():
        return None, 0, "exam_dir_missing"
    phase_paths = list_phase_paths(exam_dir)
    if not phase_paths:
        return None, 0, "no_phase_volumes"
    requested = exam_dir / f"volume_phase{phase_index}.nii.gz"
    if requested in phase_paths:
        return requested, len(phase_paths), "requested_phase"
    return phase_paths[0], len(phase_paths), f"requested_phase_missing_used_{phase_paths[0].name}"


def infer_view_from_affine(affine: np.ndarray) -> str:
    slice_vector = np.asarray(affine[:3, 2], dtype=float)
    axis = int(np.argmax(np.abs(slice_vector)))
    return ("sagittal", "coronal", "axial")[axis]


def voxel_spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    spacing = np.sqrt(np.sum(np.asarray(affine[:3, :3], dtype=float) ** 2, axis=0))
    return tuple(float(value) for value in spacing)  # type: ignore[return-value]


def robust_percentiles(data: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None, None, None
    p1, p50, p99 = np.percentile(finite, [1, 50, 99])
    return float(p1), float(p50), float(p99)


def foreground_bbox(data: np.ndarray) -> tuple[float, tuple[float, float, float] | None, tuple[float, float, float] | None]:
    finite = np.isfinite(data)
    if not finite.any():
        return float("nan"), None, None

    finite_values = data[finite]
    p1, _, p99 = np.percentile(finite_values, [1, 50, 99])
    tolerance = max(abs(float(p99)) * 1e-6, 1e-6)
    mask = finite & (np.abs(data) > tolerance)
    nonzero_fraction = float(mask.mean())

    if mask.mean() > 0.95:
        threshold = float(np.percentile(finite_values, 5))
        mask = finite & (data > threshold)
    if mask.sum() < 16:
        threshold = float(p1)
        mask = finite & (data > threshold)
    if mask.sum() < 16:
        return nonzero_fraction, None, None

    coords = np.argwhere(mask)
    mins = coords.min(axis=0).astype(float)
    maxs = coords.max(axis=0).astype(float) + 1.0
    shape = np.asarray(data.shape[:3], dtype=float)
    center = tuple(((mins + maxs) / 2.0 / shape).tolist())
    extent = tuple(((maxs - mins) / shape).tolist())
    return nonzero_fraction, center, extent  # type: ignore[return-value]


def resize_array(data: np.ndarray, output_shape: tuple[int, ...]) -> np.ndarray:
    if data.shape == output_shape:
        return data.astype(np.float32, copy=False)
    zoom = [out_size / in_size for out_size, in_size in zip(output_shape, data.shape)]
    return ndimage.zoom(data, zoom=zoom, order=1).astype(np.float32, copy=False)


def normalize_for_similarity(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape, dtype=np.float32)

    values = data[finite]
    p1, p99 = np.percentile(values, [1, 99])
    if not math.isfinite(float(p1)) or not math.isfinite(float(p99)) or p99 <= p1:
        p1 = float(np.nanmin(values))
        p99 = float(np.nanmax(values))
    if not math.isfinite(float(p1)) or not math.isfinite(float(p99)) or p99 <= p1:
        return np.zeros(data.shape, dtype=np.float32)

    normalized = (np.clip(data, p1, p99) - p1) / (p99 - p1)
    normalized[~finite] = 0.0
    return normalized.astype(np.float32, copy=False)


def make_projection_images(volume: np.ndarray, projection_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    images = []
    for axis in range(3):
        projection = np.max(volume, axis=axis)
        projection = resize_array(projection, (projection_size, projection_size))
        images.append(normalize_for_similarity(projection))
    return images[0], images[1], images[2]


def flatten_zscore(values: np.ndarray) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = np.isfinite(flat)
    if not finite.any():
        return np.zeros(flat.shape, dtype=np.float32)
    cleaned = np.zeros(flat.shape, dtype=np.float32)
    cleaned[finite] = flat[finite]
    mean = float(cleaned[finite].mean())
    std = float(cleaned[finite].std())
    if std <= 1e-8:
        return cleaned * 0.0
    cleaned[finite] = (cleaned[finite] - mean) / std
    return cleaned


def load_volume_features(
    exam_dir_text: object,
    phase_index: int,
    target_shape: tuple[int, int, int],
    projection_size: int,
) -> VolumeFeatures:
    path, phase_count, phase_selection_status = choose_phase_path(exam_dir_text, phase_index)
    if path is None:
        return VolumeFeatures(
            path="",
            phase_selection_status=phase_selection_status,
            load_error="",
            shape=None,
            canonical_shape=None,
            spacing=None,
            canonical_spacing=None,
            fov=None,
            canonical_fov=None,
            axcodes="",
            nifti_header_view="",
            phase_count=phase_count,
            nonzero_fraction=None,
            intensity_p1=None,
            intensity_p50=None,
            intensity_p99=None,
            bbox_center_fraction=None,
            bbox_extent_fraction=None,
            normalized_volume=None,
            projection_vector=None,
            projection_images=None,
        )

    try:
        image = nib.load(str(path))
        data = np.asarray(image.dataobj, dtype=np.float32)
        if data.ndim > 3:
            data = data[..., 0]
        data = np.squeeze(data)
        if data.ndim != 3:
            raise ValueError(f"expected 3D volume, got shape {data.shape}")

        affine = np.asarray(image.affine, dtype=float)
        shape = tuple(int(value) for value in data.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        fov = tuple(float(size * zoom) for size, zoom in zip(shape, spacing))
        axcodes = "".join(nib.orientations.aff2axcodes(affine))
        nifti_header_view = infer_view_from_affine(affine)
        p1, p50, p99 = robust_percentiles(data)

        canonical = nib.as_closest_canonical(image)
        canonical_data = np.asarray(canonical.dataobj, dtype=np.float32)
        if canonical_data.ndim > 3:
            canonical_data = canonical_data[..., 0]
        canonical_data = np.squeeze(canonical_data)
        if canonical_data.ndim != 3:
            raise ValueError(f"expected 3D canonical volume, got shape {canonical_data.shape}")
        canonical_shape = tuple(int(value) for value in canonical_data.shape[:3])
        canonical_spacing = voxel_spacing_from_affine(np.asarray(canonical.affine, dtype=float))
        canonical_fov = tuple(
            float(size * zoom) for size, zoom in zip(canonical_shape, canonical_spacing)
        )

        nonzero_fraction, bbox_center, bbox_extent = foreground_bbox(canonical_data)
        normalized = normalize_for_similarity(canonical_data)
        resized = resize_array(normalized, target_shape)
        projection_images = make_projection_images(resized, projection_size)
        projection_vector = np.concatenate([flatten_zscore(image_2d) for image_2d in projection_images])

        return VolumeFeatures(
            path=str(path),
            phase_selection_status=phase_selection_status,
            load_error="",
            shape=shape,
            canonical_shape=canonical_shape,
            spacing=spacing,
            canonical_spacing=canonical_spacing,
            fov=fov,
            canonical_fov=canonical_fov,
            axcodes=axcodes,
            nifti_header_view=nifti_header_view,
            phase_count=phase_count,
            nonzero_fraction=nonzero_fraction,
            intensity_p1=p1,
            intensity_p50=p50,
            intensity_p99=p99,
            bbox_center_fraction=bbox_center,
            bbox_extent_fraction=bbox_extent,
            normalized_volume=resized,
            projection_vector=projection_vector,
            projection_images=projection_images,
        )
    except Exception as exc:
        return VolumeFeatures(
            path=str(path),
            phase_selection_status=phase_selection_status,
            load_error=str(exc),
            shape=None,
            canonical_shape=None,
            spacing=None,
            canonical_spacing=None,
            fov=None,
            canonical_fov=None,
            axcodes="",
            nifti_header_view="",
            phase_count=phase_count,
            nonzero_fraction=None,
            intensity_p1=None,
            intensity_p50=None,
            intensity_p99=None,
            bbox_center_fraction=None,
            bbox_extent_fraction=None,
            normalized_volume=None,
            projection_vector=None,
            projection_images=None,
        )


def correlation(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return float("nan")
    x = flatten_zscore(left)
    y = flatten_zscore(right)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 16:
        return float("nan")
    x = x[finite]
    y = y[finite]
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator <= 1e-8:
        return float("nan")
    return float(np.dot(x, y) / denominator)


def normalized_mutual_information(left: np.ndarray | None, right: np.ndarray | None, bins: int = 48) -> float:
    if left is None or right is None:
        return float("nan")
    x = np.asarray(left, dtype=np.float32).reshape(-1)
    y = np.asarray(right, dtype=np.float32).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 32:
        return float("nan")
    x = np.clip(x[finite], 0.0, 1.0)
    y = np.clip(y[finite], 0.0, 1.0)
    hist, _, _ = np.histogram2d(x, y, bins=bins, range=((0.0, 1.0), (0.0, 1.0)))
    total = float(hist.sum())
    if total <= 0:
        return float("nan")
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nonzero = pxy > 0
    px_py = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nonzero] * np.log(pxy[nonzero] / px_py[nonzero])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx + hy <= 1e-8:
        return float("nan")
    return float(2.0 * mi / (hx + hy))


def mean_projection_ssim(
    left: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    right: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> float:
    if structural_similarity is None or left is None or right is None:
        return float("nan")
    scores = []
    for left_image, right_image in zip(left, right):
        try:
            scores.append(
                float(
                    structural_similarity(
                        np.asarray(left_image, dtype=np.float32),
                        np.asarray(right_image, dtype=np.float32),
                        data_range=1.0,
                    )
                )
            )
        except Exception:
            continue
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def max_relative_difference(
    observed: tuple[float, float, float] | None,
    reference: tuple[float, float, float] | None,
) -> float:
    if observed is None or reference is None:
        return float("nan")
    diffs = []
    for obs, ref in zip(observed, reference):
        denominator = max(abs(float(ref)), 1e-6)
        diffs.append(abs(float(obs) - float(ref)) / denominator)
    return float(max(diffs))


def euclidean_distance(
    observed: tuple[float, float, float] | None,
    reference: tuple[float, float, float] | None,
) -> float:
    if observed is None or reference is None:
        return float("nan")
    return float(np.linalg.norm(np.asarray(observed, dtype=float) - np.asarray(reference, dtype=float)))


def corr_to_unit_interval(value: float) -> float:
    if not math.isfinite(value):
        return float("nan")
    return min(max((value + 1.0) / 2.0, 0.0), 1.0)


def visual_consistency_score(volume_corr: float, mip_corr: float, nmi: float, ssim_mip: float) -> float:
    parts = []
    weights = []
    for value, weight, is_corr in (
        (mip_corr, 0.45, True),
        (volume_corr, 0.25, True),
        (ssim_mip, 0.20, False),
        (nmi, 0.10, False),
    ):
        if not math.isfinite(value):
            continue
        if is_corr:
            value = corr_to_unit_interval(value)
        else:
            value = min(max(value, 0.0), 1.0)
        parts.append(value * weight)
        weights.append(weight)
    if not weights:
        return float("nan")
    return float(sum(parts) / sum(weights))


def choose_reference(patient_df: pd.DataFrame) -> tuple[int, str]:
    anchors = patient_df[patient_df["is_anchor_date"].map(parse_bool)]
    if not anchors.empty:
        return int(anchors.index[0]), "active_anchor"
    sorted_patient = patient_df.assign(
        _exam_datetime=patient_df["exam_date"].map(parse_exam_datetime)
    ).sort_values(["_exam_datetime", "exam_date"], kind="stable")
    return int(sorted_patient.index[0]), "earliest_active_fallback"


def status_for_comparison(row: dict[str, object], args: argparse.Namespace) -> tuple[str, str]:
    if row["is_reference_exam"]:
        return "reference_exam", ""
    issues = []
    if row["load_error"]:
        issues.append("volume_load_error")
    if row["reference_load_error"]:
        issues.append("reference_load_error")
    if row["nifti_header_view_matches_reference"] is False:
        issues.append("nifti_header_view_mismatch")
    fov_rel_diff = row["fov_rel_diff_max"]
    if isinstance(fov_rel_diff, float) and math.isfinite(fov_rel_diff):
        if fov_rel_diff > args.fov_rel_threshold:
            issues.append("fov_difference")
    spacing_rel_diff = row["spacing_rel_diff_max"]
    if isinstance(spacing_rel_diff, float) and math.isfinite(spacing_rel_diff):
        if spacing_rel_diff > args.spacing_rel_threshold:
            issues.append("spacing_difference")
    visual_score = row["visual_consistency_score"]
    mip_corr = row["mip_corr"]
    volume_corr = row["volume_corr"]
    if (
        isinstance(visual_score, float)
        and math.isfinite(visual_score)
        and visual_score < args.visual_score_threshold
        and (
            not isinstance(mip_corr, float)
            or not math.isfinite(mip_corr)
            or mip_corr < args.mip_corr_threshold
        )
        and (
            not isinstance(volume_corr, float)
            or not math.isfinite(volume_corr)
            or volume_corr < args.volume_corr_threshold
        )
    ):
        issues.append("low_visual_similarity")
    if not issues:
        return "ok", ""
    unique_issues = sorted(set(issues))
    return "review_recommended", "|".join(unique_issues)


def feature_record(prefix: str, features: VolumeFeatures) -> dict[str, object]:
    return {
        f"{prefix}volume_path": features.path,
        f"{prefix}phase_selection_status": features.phase_selection_status,
        f"{prefix}phase_count": features.phase_count,
        f"{prefix}load_error": features.load_error,
        f"{prefix}shape": tuple_to_text(features.shape),
        f"{prefix}canonical_shape": tuple_to_text(features.canonical_shape),
        f"{prefix}spacing": tuple_to_text(features.spacing),
        f"{prefix}canonical_spacing": tuple_to_text(features.canonical_spacing),
        f"{prefix}fov_mm": tuple_to_text(features.fov),
        f"{prefix}canonical_fov_mm": tuple_to_text(features.canonical_fov),
        f"{prefix}axcodes": features.axcodes,
        f"{prefix}nifti_header_view": features.nifti_header_view,
        f"{prefix}nonzero_fraction": safe_float(features.nonzero_fraction),
        f"{prefix}intensity_p1": safe_float(features.intensity_p1),
        f"{prefix}intensity_p50": safe_float(features.intensity_p50),
        f"{prefix}intensity_p99": safe_float(features.intensity_p99),
        f"{prefix}bbox_center_fraction": tuple_to_text(features.bbox_center_fraction),
        f"{prefix}bbox_extent_fraction": tuple_to_text(features.bbox_extent_fraction),
    }


def compare_features(features: VolumeFeatures, reference: VolumeFeatures) -> dict[str, object]:
    volume_corr = correlation(features.normalized_volume, reference.normalized_volume)
    mip_corr = correlation(features.projection_vector, reference.projection_vector)
    nmi = normalized_mutual_information(features.normalized_volume, reference.normalized_volume)
    ssim_mip = mean_projection_ssim(features.projection_images, reference.projection_images)
    score = visual_consistency_score(volume_corr, mip_corr, nmi, ssim_mip)
    return {
        "nifti_header_view_matches_reference": (
            ""
            if not features.nifti_header_view or not reference.nifti_header_view
            else features.nifti_header_view == reference.nifti_header_view
        ),
        "shape_matches_reference": (
            "" if features.canonical_shape is None or reference.canonical_shape is None else features.canonical_shape == reference.canonical_shape
        ),
        "phase_count_matches_reference": features.phase_count == reference.phase_count,
        "spacing_rel_diff_max": max_relative_difference(
            features.canonical_spacing, reference.canonical_spacing
        ),
        "fov_rel_diff_max": max_relative_difference(features.canonical_fov, reference.canonical_fov),
        "bbox_center_distance": euclidean_distance(
            features.bbox_center_fraction, reference.bbox_center_fraction
        ),
        "bbox_extent_rel_diff_max": max_relative_difference(
            features.bbox_extent_fraction, reference.bbox_extent_fraction
        ),
        "volume_corr": volume_corr,
        "mip_corr": mip_corr,
        "normalized_mutual_information": nmi,
        "projection_ssim_mean": ssim_mip,
        "visual_consistency_score": score,
    }


def audit(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target_shape = parse_target_shape(args.target_shape)
    audit_df = pd.read_csv(args.reconstruction_audit_csv)
    active_df = audit_df[audit_df["reconstructed_exam_exists"].map(parse_bool)].copy()
    active_df = active_df.sort_values(["dataset", "patient_id", "exam_date"], kind="stable")
    if args.max_patients is not None:
        patient_ids = active_df["patient_id"].drop_duplicates().head(args.max_patients)
        active_df = active_df[active_df["patient_id"].isin(set(patient_ids))].copy()

    features_by_index: dict[int, VolumeFeatures] = {}
    for index, row in active_df.iterrows():
        features_by_index[int(index)] = load_volume_features(
            row.get("reconstructed_exam_dir", ""),
            phase_index=args.phase_index,
            target_shape=target_shape,
            projection_size=args.projection_size,
        )

    exam_rows = []
    for patient_id, patient_df in active_df.groupby("patient_id", sort=False):
        reference_index, reference_kind = choose_reference(patient_df)
        reference_row = active_df.loc[reference_index]
        reference_features = features_by_index[reference_index]

        for index, row in patient_df.iterrows():
            features = features_by_index[int(index)]
            comparison = compare_features(features, reference_features)
            record = {
                "patient_id": patient_id,
                "dataset": row.get("dataset", ""),
                "exam_date": row.get("exam_date", ""),
                "is_anchor_date": parse_bool(row.get("is_anchor_date", "")),
                "is_reference_exam": int(index) == reference_index,
                "reference_exam_date": reference_row.get("exam_date", ""),
                "reference_kind": reference_kind,
                "reconstructed_exam_dir": row.get("reconstructed_exam_dir", ""),
                "risk_status_from_reconstruction_audit": row.get("risk_status", ""),
                "selection_note": row.get("selection_note", ""),
            }
            record.update(feature_record("", features))
            record.update(feature_record("reference_", reference_features))
            record.update(comparison)
            status, reason = status_for_comparison(record, args)
            record["visual_consistency_status"] = status
            record["review_reason"] = reason
            exam_rows.append(record)

    exam_df = pd.DataFrame(exam_rows)
    patient_rows = []
    for patient_id, patient_df in exam_df.groupby("patient_id", sort=False):
        non_reference = patient_df[~patient_df["is_reference_exam"]]
        review = non_reference[non_reference["visual_consistency_status"] != "ok"]
        patient_rows.append(
            {
                "patient_id": patient_id,
                "dataset": patient_df["dataset"].iloc[0],
                "num_active_exams": int(len(patient_df)),
                "num_extra_exams": int(len(non_reference)),
                "reference_exam_date": patient_df["reference_exam_date"].iloc[0],
                "reference_kind": patient_df["reference_kind"].iloc[0],
                "anchor_exam_active": bool(patient_df["is_anchor_date"].any()),
                "num_review_exams": int(len(review)),
                "review_exam_dates": " | ".join(str(value) for value in review["exam_date"].tolist()),
                "review_reasons": " | ".join(
                    sorted({str(value) for value in review["review_reason"].tolist() if str(value)})
                ),
                "min_visual_consistency_score": safe_float(
                    pd.to_numeric(non_reference["visual_consistency_score"], errors="coerce").min()
                    if not non_reference.empty
                    else 1.0
                ),
                "min_mip_corr": safe_float(
                    pd.to_numeric(non_reference["mip_corr"], errors="coerce").min()
                    if not non_reference.empty
                    else 1.0
                ),
                "min_volume_corr": safe_float(
                    pd.to_numeric(non_reference["volume_corr"], errors="coerce").min()
                    if not non_reference.empty
                    else 1.0
                ),
                "max_fov_rel_diff": safe_float(
                    pd.to_numeric(non_reference["fov_rel_diff_max"], errors="coerce").max()
                    if not non_reference.empty
                    else 0.0
                ),
                "all_nifti_header_views_match_reference": bool(
                    (patient_df["nifti_header_view_matches_reference"] != False).all()
                ),
                "all_phase_counts_match_reference": bool(
                    (patient_df["phase_count_matches_reference"] != False).all()
                ),
            }
        )
    patient_summary_df = pd.DataFrame(patient_rows)

    flagged = exam_df[exam_df["visual_consistency_status"] == "review_recommended"].copy()
    if not flagged.empty:
        flagged["review_rank_key"] = pd.to_numeric(
            flagged["visual_consistency_score"], errors="coerce"
        ).fillna(-1.0)
        flagged = flagged.sort_values(
            ["review_rank_key", "patient_id", "exam_date"],
            ascending=[True, True, True],
            kind="stable",
        ).drop(columns=["review_rank_key"])

    summary = {
        "reconstruction_audit_csv": str(args.reconstruction_audit_csv.resolve()),
        "phase_index": int(args.phase_index),
        "target_shape": args.target_shape,
        "projection_size": int(args.projection_size),
        "max_patients": args.max_patients,
        "exam_rows": int(len(exam_df)),
        "patient_rows": int(len(patient_summary_df)),
        "review_recommended_exam_rows": int(len(flagged)),
        "status_counts": compact_counts(exam_df.get("visual_consistency_status", [])),
        "review_reason_counts": compact_counts(
            reason
            for reasons in exam_df.get("review_reason", [])
            for reason in str(reasons).split("|")
            if reason
        ),
        "reference_kind_counts": compact_counts(exam_df.get("reference_kind", [])),
        "nifti_header_view_counts": compact_counts(exam_df.get("nifti_header_view", [])),
        "nifti_header_view_matches_reference_counts": compact_counts(
            exam_df.get("nifti_header_view_matches_reference", [])
        ),
        "phase_count_matches_reference_counts": compact_counts(
            exam_df.get("phase_count_matches_reference", [])
        ),
        "patients_with_review_recommended": int((patient_summary_df["num_review_exams"] > 0).sum()),
    }
    return exam_df, patient_summary_df, flagged, summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    exam_df, patient_df, flagged_df, summary = audit(args)

    exam_path = args.output_root / "volume_visual_exam_audit.csv"
    patient_path = args.output_root / "volume_visual_patient_summary.csv"
    flagged_path = args.output_root / "volume_visual_review_queue.csv"
    summary_path = args.output_root / "summary.json"
    exam_df.to_csv(exam_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    flagged_df.to_csv(flagged_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Exam rows:       {summary['exam_rows']}")
    print(f"Patient rows:    {summary['patient_rows']}")
    print(f"Review exams:    {summary['review_recommended_exam_rows']}")
    print("Status counts:")
    for key, value in summary["status_counts"].items():
        print(f"  {key}: {value}")
    print("Review reasons:")
    for key, value in summary["review_reason_counts"].items():
        print(f"  {key}: {value}")
    print(f"Exam audit:      {exam_path}")
    print(f"Patient audit:   {patient_path}")
    print(f"Review queue:    {flagged_path}")
    print(f"Summary:         {summary_path}")


if __name__ == "__main__":
    main()
