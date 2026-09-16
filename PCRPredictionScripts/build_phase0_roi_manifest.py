#!/usr/bin/env python3
"""Build a phase-0 tumor ROI manifest for pCR classification.

The manifest uses reconstructed anchor phase-0 images and nnU-Net predictions
from the MAMA-MIA segmentation QC run. It applies simple quality gates before
turning a predicted mask into a fixed physical crop around the tumor center.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QC_REPORT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "mamamia_nnunet_qc_phase0_anchor"
    / "nnunet_reconstructed_vs_mamamia_qc_report.csv"
)
DEFAULT_CLINICAL_INFO = (
    PROJECT_ROOT / "Original_Datasets" / "MAMA-MIA" / "clinical_and_imaging_info.xlsx"
)
DEFAULT_SPLITS = PROJECT_ROOT / "Original_Datasets" / "MAMA-MIA" / "train_test_splits.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "phase0_nnunet_roi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pCR classification ROI manifest from phase-0 reconstructed "
            "images and MAMA-MIA nnU-Net tumor predictions."
        )
    )
    parser.add_argument("--qc-report", type=Path, default=DEFAULT_QC_REPORT)
    parser.add_argument("--clinical-info", type=Path, default=DEFAULT_CLINICAL_INFO)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--crop-size-mm",
        type=float,
        nargs=3,
        default=(96.0, 96.0, 64.0),
        metavar=("X", "Y", "Z"),
        help="Fixed physical crop size around the tumor center.",
    )
    parser.add_argument(
        "--min-volume-ratio",
        type=float,
        default=0.2,
        help="Minimum predicted-mask/MAMA-MIA-mask volume ratio for accepting a mask.",
    )
    parser.add_argument(
        "--max-volume-ratio",
        type=float,
        default=5.0,
        help="Maximum predicted-mask/MAMA-MIA-mask volume ratio for accepting a mask.",
    )
    parser.add_argument(
        "--min-pred-volume-ml",
        type=float,
        default=0.05,
        help="Reject tiny non-empty predictions below this volume.",
    )
    parser.add_argument(
        "--component-mode",
        choices=("largest", "all"),
        default="all",
        help="Use the largest connected component or all positive voxels for the ROI center.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for syntax/smoke checks.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Print progress every N rows while building the manifest.",
    )
    parser.add_argument(
        "--include-status",
        nargs="+",
        default=("nnunet_pass",),
        help="ROI statuses marked usable_for_training.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_clinical(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name="dataset_info")
    if "patient_id" not in clinical or "pcr" not in clinical:
        raise ValueError(f"{path} must contain patient_id and pcr columns")
    clinical = clinical.copy()
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    clinical["pcr"] = pd.to_numeric(clinical["pcr"], errors="coerce")
    return clinical


def load_splits(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    records: list[dict[str, str]] = []
    for column, split_name in (("train_split", "train"), ("test_split", "test")):
        if column not in split_df:
            continue
        for patient_id in split_df[column].dropna().astype(str):
            records.append({"patient_id": patient_id, "split": split_name})
    return pd.DataFrame.from_records(records).drop_duplicates("patient_id")


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, num_labels = ndimage.label(mask, structure=structure)
    if num_labels == 0:
        return mask, 0, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(sizes.argmax())
    component = labels == label_id
    return component, int(component.sum()), int(num_labels)


def mask_center_and_bbox(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot compute center/bbox for an empty mask")
    return coords.mean(axis=0), coords.min(axis=0), coords.max(axis=0) + 1


def fixed_crop_for_center(
    center_vox: np.ndarray,
    image_shape: Iterable[int],
    spacing: Iterable[float],
    crop_size_mm: Iterable[float],
) -> dict[str, object]:
    shape = np.asarray(tuple(image_shape), dtype=int)
    spacing_arr = np.asarray(tuple(spacing), dtype=float)
    crop_mm = np.asarray(tuple(crop_size_mm), dtype=float)
    crop_vox = np.maximum(1, np.round(crop_mm / spacing_arr).astype(int))
    start = np.floor(center_vox - crop_vox / 2.0).astype(int)
    end = start + crop_vox
    clipped_start = np.maximum(start, 0)
    clipped_end = np.minimum(end, shape)
    needs_padding = bool(np.any(start < 0) or np.any(end > shape))
    return {
        "crop_size_mm_x": float(crop_mm[0]),
        "crop_size_mm_y": float(crop_mm[1]),
        "crop_size_mm_z": float(crop_mm[2]),
        "crop_size_vox_x": int(crop_vox[0]),
        "crop_size_vox_y": int(crop_vox[1]),
        "crop_size_vox_z": int(crop_vox[2]),
        "crop_start_x": int(start[0]),
        "crop_start_y": int(start[1]),
        "crop_start_z": int(start[2]),
        "crop_end_x": int(end[0]),
        "crop_end_y": int(end[1]),
        "crop_end_z": int(end[2]),
        "crop_clipped_start_x": int(clipped_start[0]),
        "crop_clipped_start_y": int(clipped_start[1]),
        "crop_clipped_start_z": int(clipped_start[2]),
        "crop_clipped_end_x": int(clipped_end[0]),
        "crop_clipped_end_y": int(clipped_end[1]),
        "crop_clipped_end_z": int(clipped_end[2]),
        "crop_needs_padding": needs_padding,
    }


def volume_ratio_status(
    pred_volume_ml: float,
    reference_volume_ml: float,
    min_ratio: float,
    max_ratio: float,
    min_pred_volume_ml: float,
) -> tuple[str, float]:
    if not np.isfinite(pred_volume_ml) or pred_volume_ml <= 0:
        return "empty_prediction", 0.0
    if pred_volume_ml < min_pred_volume_ml:
        return "pred_volume_too_small", np.nan
    if not np.isfinite(reference_volume_ml) or reference_volume_ml <= 0:
        return "missing_reference_volume", np.nan
    ratio = pred_volume_ml / reference_volume_ml
    if ratio < min_ratio:
        return "bad_volume_ratio_low", ratio
    if ratio > max_ratio:
        return "bad_volume_ratio_high", ratio
    return "nnunet_pass", ratio


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    qc = pd.read_csv(args.qc_report)
    if args.max_rows is not None:
        qc = qc.head(args.max_rows).copy()
    clinical = load_clinical(args.clinical_info)
    splits = load_splits(args.splits)

    merged = qc.merge(clinical, on="patient_id", how="left", suffixes=("", "_clinical"))
    merged = merged.merge(splits, on="patient_id", how="left")

    records: list[dict[str, object]] = []
    include_status = set(args.include_status)

    for row_idx, (_, row) in enumerate(merged.iterrows(), start=1):
        if args.progress_every and row_idx % args.progress_every == 0:
            print(f"Processed {row_idx}/{len(merged)} rows", flush=True)
        record = row.to_dict()
        image_path = resolve_path(row["reconstructed_image_path"])
        pred_path = resolve_path(row["reconstructed_prediction_path"])
        record["reconstructed_image_path"] = str(image_path)
        record["reconstructed_prediction_path"] = str(pred_path)
        record["roi_source"] = "nnunet_prediction"

        pred_volume = float(row.get("reconstructed_pred_volume_ml", np.nan))
        ref_volume = float(row.get("mamamia_mask_volume_ml", np.nan))
        status, ratio = volume_ratio_status(
            pred_volume,
            ref_volume,
            args.min_volume_ratio,
            args.max_volume_ratio,
            args.min_pred_volume_ml,
        )
        record["roi_status"] = status
        record["volume_ratio_prediction_to_mamamia"] = ratio

        if not image_path.exists():
            record["roi_status"] = "missing_reconstructed_image"
        elif not pred_path.exists():
            record["roi_status"] = "missing_prediction"
        elif pd.isna(row.get("pcr")):
            record["roi_status"] = "missing_pcr_label"

        if record["roi_status"] == "nnunet_pass":
            try:
                image = nib.load(str(image_path))
                pred = nib.load(str(pred_path))
                mask = np.asanyarray(pred.dataobj) > 0
                if mask.shape != image.shape:
                    record["roi_status"] = "prediction_image_shape_mismatch"
                else:
                    full_voxels = int(mask.sum())
                    component_voxels = full_voxels
                    num_components = np.nan
                    if args.component_mode == "largest":
                        mask, component_voxels, num_components = largest_component(mask)
                    center, bbox_start, bbox_end = mask_center_and_bbox(mask)
                    spacing = image.header.get_zooms()[:3]
                    voxel_volume_ml = float(abs(np.linalg.det(image.affine[:3, :3])) / 1000.0)
                    record.update(
                        {
                            "image_shape_x": int(image.shape[0]),
                            "image_shape_y": int(image.shape[1]),
                            "image_shape_z": int(image.shape[2]),
                            "image_spacing_x": float(spacing[0]),
                            "image_spacing_y": float(spacing[1]),
                            "image_spacing_z": float(spacing[2]),
                            "mask_full_voxels": full_voxels,
                            "mask_component_voxels": component_voxels,
                            "mask_num_components": num_components,
                            "mask_component_fraction": (
                                component_voxels / full_voxels if full_voxels else np.nan
                            ),
                            "mask_component_volume_ml": component_voxels * voxel_volume_ml,
                            "center_vox_x": float(center[0]),
                            "center_vox_y": float(center[1]),
                            "center_vox_z": float(center[2]),
                            "bbox_start_x": int(bbox_start[0]),
                            "bbox_start_y": int(bbox_start[1]),
                            "bbox_start_z": int(bbox_start[2]),
                            "bbox_end_x": int(bbox_end[0]),
                            "bbox_end_y": int(bbox_end[1]),
                            "bbox_end_z": int(bbox_end[2]),
                        }
                    )
                    record.update(
                        fixed_crop_for_center(center, image.shape, spacing, args.crop_size_mm)
                    )
            except Exception as exc:  # noqa: BLE001 - record the case and keep building the manifest.
                record["roi_status"] = "mask_read_or_geometry_error"
                record["roi_error"] = str(exc)

        record["usable_for_training"] = record["roi_status"] in include_status
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_root / "roi_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    usable = manifest[manifest["usable_for_training"]]
    summary = {
        "qc_report": str(args.qc_report.resolve()),
        "clinical_info": str(args.clinical_info.resolve()),
        "splits": str(args.splits.resolve()),
        "crop_size_mm": list(map(float, args.crop_size_mm)),
        "min_volume_ratio": args.min_volume_ratio,
        "max_volume_ratio": args.max_volume_ratio,
        "min_pred_volume_ml": args.min_pred_volume_ml,
        "component_mode": args.component_mode,
        "rows": int(len(manifest)),
        "usable_rows": int(len(usable)),
        "roi_status_counts": {
            str(k): int(v)
            for k, v in manifest["roi_status"].value_counts(dropna=False).items()
        },
        "usable_split_counts": {
            str(k): int(v) for k, v in usable["split"].value_counts(dropna=False).items()
        },
        "usable_pcr_counts": {
            str(int(k)): int(v)
            for k, v in usable["pcr"].dropna().astype(int).value_counts().sort_index().items()
        },
        "usable_dataset_counts": {
            str(k): int(v) for k, v in usable["dataset"].value_counts(dropna=False).items()
        },
    }
    summary_path = output_root / "roi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
