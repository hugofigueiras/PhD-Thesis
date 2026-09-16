#!/usr/bin/env python3
"""Build a MAMA-MIA multiphase manifest for foundation-model pCR benchmarking.

The manifest is one row per patient. It records the original MAMA-MIA DCE phase
paths, official split, pCR label, expert-mask ROI crop coordinates, and benchmark
eligibility flags. It does not write image crops, subtraction images, embeddings,
or any preprocessed tensors to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "manifests"
DEFAULT_MANIFEST_NAME = "mamamia_multiphase_foundation_manifest.csv"
DEFAULT_SUMMARY_NAME = "mamamia_multiphase_foundation_summary.json"

ALLOWED_BASELINE_CLINICAL_COLUMNS = [
    "age",
    "menopause",
    "er",
    "pr",
    "hr",
    "her2",
    "tumor_subtype",
    "nottingham_grade",
    "mammaprint",
    "oncotype_score",
    "bilateral_breast_cancer",
    "multifocal_cancer",
    "ethnicity",
    "has_implant",
    "bmi_group",
    "breast_density",
]

DISCUSS_BEFORE_USING_CLINICAL_COLUMNS = [
    "nac_agent",
    "endocrine_therapy",
    "anti_her2_neu_therapy",
]

EXCLUDED_CLINICAL_COLUMNS = [
    "mastectomy_post_nac",
    "days_to_follow_up",
    "days_to_recurrence",
    "days_to_metastasis",
    "days_to_death",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a patient-level MAMA-MIA multiphase manifest for frozen "
            "foundation-model pCR benchmarks."
        )
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--clinical-info", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-name", default=DEFAULT_MANIFEST_NAME)
    parser.add_argument("--summary-name", default=DEFAULT_SUMMARY_NAME)
    parser.add_argument(
        "--roi-margin-mm",
        type=float,
        nargs=3,
        default=(30.0, 30.0, 20.0),
        metavar=("X", "Y", "Z"),
        help="Physical context margin added around the expert-mask bounding box.",
    )
    parser.add_argument(
        "--strict-affine",
        action="store_true",
        help="Reject expert ROI if the expert mask affine differs from phase 0.",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-3)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for lightweight smoke checks.",
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


def available_phase_paths(mamamia_root: Path, patient_id: str) -> dict[int, Path]:
    image_dir = mamamia_root / "images" / patient_id
    if not image_dir.exists():
        return {}

    prefix = f"{patient_id}_"
    phase_paths: dict[int, Path] = {}
    for path in sorted(image_dir.glob(f"{patient_id}_*.nii.gz")):
        phase_text = path.name.removeprefix(prefix).removesuffix(".nii.gz")
        if phase_text.isdigit():
            phase_paths[int(phase_text)] = path
    return phase_paths


def image_header_record(path: Path) -> dict[str, object]:
    image = nib.load(str(path))
    spacing = image.header.get_zooms()[:3]
    return {
        "shape": tuple(int(v) for v in image.shape[:3]),
        "spacing": tuple(float(v) for v in spacing),
        "dtype": str(image.get_data_dtype()),
        "affine": image.affine,
    }


def add_phase_record(
    record: dict[str, object],
    role: str,
    index: int | None,
    path: Path | None,
    header: dict[str, object] | None,
    error: str | None,
) -> None:
    record[f"{role}_index"] = np.nan if index is None else int(index)
    record[f"{role}_path"] = "" if path is None else str(path)
    record[f"{role}_exists"] = bool(path is not None and path.exists())
    record[f"{role}_header_error"] = "" if error is None else error
    if header is None:
        return

    shape = header["shape"]
    spacing = header["spacing"]
    record.update(
        {
            f"{role}_shape_x": int(shape[0]),
            f"{role}_shape_y": int(shape[1]),
            f"{role}_shape_z": int(shape[2]),
            f"{role}_spacing_x": float(spacing[0]),
            f"{role}_spacing_y": float(spacing[1]),
            f"{role}_spacing_z": float(spacing[2]),
            f"{role}_dtype": str(header["dtype"]),
        }
    )


def phase_geometry_status(
    headers: Iterable[dict[str, object]],
    affine_atol: float,
) -> str:
    header_list = list(headers)
    if not header_list:
        return "missing_headers"

    first = header_list[0]
    first_shape = first["shape"]
    first_affine = first["affine"]
    for header in header_list[1:]:
        if header["shape"] != first_shape:
            return "shape_mismatch"
        if not np.allclose(header["affine"], first_affine, atol=affine_atol):
            return "affine_mismatch"
    return "matched"


def expert_mask_path(mamamia_root: Path, patient_id: str) -> Path:
    return mamamia_root / "segmentations" / "expert" / f"{patient_id}.nii.gz"


def automatic_mask_path(mamamia_root: Path, patient_id: str) -> Path:
    return mamamia_root / "segmentations" / "automatic" / f"{patient_id}.nii.gz"


def expert_roi_record(
    image_path: Path,
    mask_path: Path,
    margin_mm: Iterable[float],
    strict_affine: bool,
    affine_atol: float,
) -> dict[str, object]:
    image = nib.load(str(image_path))
    mask_img = nib.load(str(mask_path))
    mask = np.asanyarray(mask_img.dataobj) > 0

    affine_matches = bool(np.allclose(image.affine, mask_img.affine, atol=affine_atol))
    if mask.shape != image.shape[:3]:
        return {
            "expert_roi_status": "mask_image_shape_mismatch",
            "expert_roi_affine_matches_phase0": affine_matches,
        }
    if strict_affine and not affine_matches:
        return {
            "expert_roi_status": "mask_image_affine_mismatch",
            "expert_roi_affine_matches_phase0": affine_matches,
        }
    if not mask.any():
        return {
            "expert_roi_status": "empty_mask",
            "expert_roi_affine_matches_phase0": affine_matches,
        }

    coords = np.argwhere(mask)
    shape = np.asarray(image.shape[:3], dtype=int)
    spacing = np.asarray(image.header.get_zooms()[:3], dtype=float)
    margin_arr = np.asarray(tuple(margin_mm), dtype=float)
    margin_vox = np.ceil(margin_arr / spacing).astype(int)

    bbox_start = coords.min(axis=0)
    bbox_end = coords.max(axis=0) + 1
    crop_start = bbox_start - margin_vox
    crop_end = bbox_end + margin_vox
    clipped_start = np.maximum(crop_start, 0)
    clipped_end = np.minimum(crop_end, shape)
    center = coords.mean(axis=0)
    voxel_volume_ml = float(abs(np.linalg.det(image.affine[:3, :3])) / 1000.0)

    return {
        "expert_roi_status": "expert_mask_roi",
        "expert_roi_affine_matches_phase0": affine_matches,
        "expert_roi_mask_voxels": int(mask.sum()),
        "expert_roi_mask_volume_ml": float(mask.sum() * voxel_volume_ml),
        "expert_roi_center_vox_x": float(center[0]),
        "expert_roi_center_vox_y": float(center[1]),
        "expert_roi_center_vox_z": float(center[2]),
        "expert_roi_bbox_start_x": int(bbox_start[0]),
        "expert_roi_bbox_start_y": int(bbox_start[1]),
        "expert_roi_bbox_start_z": int(bbox_start[2]),
        "expert_roi_bbox_end_x": int(bbox_end[0]),
        "expert_roi_bbox_end_y": int(bbox_end[1]),
        "expert_roi_bbox_end_z": int(bbox_end[2]),
        "expert_roi_margin_mm_x": float(margin_arr[0]),
        "expert_roi_margin_mm_y": float(margin_arr[1]),
        "expert_roi_margin_mm_z": float(margin_arr[2]),
        "expert_roi_margin_vox_x": int(margin_vox[0]),
        "expert_roi_margin_vox_y": int(margin_vox[1]),
        "expert_roi_margin_vox_z": int(margin_vox[2]),
        "expert_roi_crop_start_x": int(crop_start[0]),
        "expert_roi_crop_start_y": int(crop_start[1]),
        "expert_roi_crop_start_z": int(crop_start[2]),
        "expert_roi_crop_end_x": int(crop_end[0]),
        "expert_roi_crop_end_y": int(crop_end[1]),
        "expert_roi_crop_end_z": int(crop_end[2]),
        "expert_roi_crop_clipped_start_x": int(clipped_start[0]),
        "expert_roi_crop_clipped_start_y": int(clipped_start[1]),
        "expert_roi_crop_clipped_start_z": int(clipped_start[2]),
        "expert_roi_crop_clipped_end_x": int(clipped_end[0]),
        "expert_roi_crop_clipped_end_y": int(clipped_end[1]),
        "expert_roi_crop_clipped_end_z": int(clipped_end[2]),
        "expert_roi_crop_needs_padding": bool(
            np.any(crop_start < 0) or np.any(crop_end > shape)
        ),
    }


def compact_counts(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).items()}


def subset_counts(manifest: pd.DataFrame, flag: str, column: str) -> dict[str, int]:
    usable = manifest[manifest[flag]]
    return compact_counts(usable[column]) if column in usable else {}


def numeric_summary(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def summarize_manifest(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    clinical_info: Path,
    splits: Path,
    manifest_path: Path,
) -> dict[str, object]:
    usable_whole = manifest[manifest["usable_for_whole_volume_benchmark"]]
    usable_multiphase = manifest[manifest["usable_for_multiphase_or_subtraction_benchmark"]]
    usable_roi = manifest[manifest["usable_for_expert_roi_benchmark"]]

    return {
        "mamamia_root": str(args.mamamia_root.resolve()),
        "clinical_info": str(clinical_info),
        "splits": str(splits),
        "manifest": str(manifest_path),
        "rows": int(len(manifest)),
        "usable_for_whole_volume_benchmark": int(len(usable_whole)),
        "usable_for_multiphase_or_subtraction_benchmark": int(len(usable_multiphase)),
        "usable_for_expert_roi_benchmark": int(len(usable_roi)),
        "benchmark_status_counts": compact_counts(manifest["benchmark_status"]),
        "phase_geometry_status_counts": compact_counts(manifest["phase_geometry_status"]),
        "expert_roi_status_counts": compact_counts(manifest["expert_roi_status"]),
        "num_available_phases_counts": {
            str(int(k)): int(v)
            for k, v in manifest["num_available_phases"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .items()
        },
        "whole_volume_split_counts": subset_counts(
            manifest, "usable_for_whole_volume_benchmark", "split"
        ),
        "whole_volume_pcr_counts": subset_counts(
            manifest, "usable_for_whole_volume_benchmark", "pcr"
        ),
        "whole_volume_dataset_counts": subset_counts(
            manifest, "usable_for_whole_volume_benchmark", "dataset"
        ),
        "expert_roi_split_counts": subset_counts(
            manifest, "usable_for_expert_roi_benchmark", "split"
        ),
        "expert_roi_pcr_counts": subset_counts(
            manifest, "usable_for_expert_roi_benchmark", "pcr"
        ),
        "expert_roi_dataset_counts": subset_counts(
            manifest, "usable_for_expert_roi_benchmark", "dataset"
        ),
        "expert_roi_mask_volume_ml": numeric_summary(
            manifest.get("expert_roi_mask_volume_ml", pd.Series(dtype=float))
        ),
        "required_input_roles": [
            "phase0",
            "phase1",
            "phase2",
            "last_phase",
            "phase1_minus_phase0",
            "phase2_minus_phase0",
            "last_phase_minus_phase0",
        ],
        "clinical_feature_policy": {
            "allowed_baseline_columns": [
                c for c in ALLOWED_BASELINE_CLINICAL_COLUMNS if c in manifest.columns
            ],
            "discuss_before_using_columns": [
                c
                for c in DISCUSS_BEFORE_USING_CLINICAL_COLUMNS
                if c in manifest.columns
            ],
            "excluded_leakage_columns": [
                c for c in EXCLUDED_CLINICAL_COLUMNS if c in manifest.columns
            ],
        },
        "notes": [
            "No image crops or subtraction images are written by this script.",
            "Subtraction eligibility requires phase 0, phase 1, phase 2, and last phase geometry to match.",
            "Expert ROI uses the phase-0 expert mask crop and assumes phase geometry is matched for multiphase ROI extraction.",
            "Use frozen foundation-model embeddings first; full fine-tuning is a later experiment.",
        ],
    }


def patient_record(
    row: pd.Series,
    mamamia_root: Path,
    roi_margin_mm: Iterable[float],
    strict_affine: bool,
    affine_atol: float,
) -> dict[str, object]:
    patient_id = str(row["patient_id"])
    record = row.to_dict()

    phase_paths = available_phase_paths(mamamia_root, patient_id)
    phase_indices = sorted(phase_paths)
    last_index = phase_indices[-1] if phase_indices else None
    required_roles = {
        "phase0": 0,
        "phase1": 1,
        "phase2": 2,
        "last_phase": last_index,
    }

    record.update(
        {
            "num_available_phases": int(len(phase_indices)),
            "available_phase_indices": ";".join(str(i) for i in phase_indices),
            "available_phase_paths": ";".join(str(phase_paths[i]) for i in phase_indices),
            "expert_mask_path": str(expert_mask_path(mamamia_root, patient_id)),
            "automatic_mask_path": str(automatic_mask_path(mamamia_root, patient_id)),
        }
    )
    record["expert_mask_exists"] = Path(str(record["expert_mask_path"])).exists()
    record["automatic_mask_exists"] = Path(str(record["automatic_mask_path"])).exists()

    headers_by_role: dict[str, dict[str, object]] = {}
    missing_roles: list[str] = []
    header_errors: list[str] = []
    for role, index in required_roles.items():
        path = None if index is None else phase_paths.get(index)
        header = None
        error = None
        if path is None:
            missing_roles.append(role)
        else:
            try:
                header = image_header_record(path)
                headers_by_role[role] = header
            except Exception as exc:  # noqa: BLE001 - manifest should record bad rows.
                error = str(exc)
                header_errors.append(f"{role}: {exc}")
        add_phase_record(record, role, index, path, header, error)

    if missing_roles:
        geometry = "missing_required_phase"
    elif header_errors:
        geometry = "phase_header_error"
    else:
        # Last phase can duplicate phase2 for 3-phase patients; comparing twice is harmless.
        geometry = phase_geometry_status(headers_by_role.values(), affine_atol)
    record["phase_geometry_status"] = geometry
    record["missing_required_phase_roles"] = ";".join(missing_roles)
    record["phase_header_errors"] = ";".join(header_errors)

    phase0_path = Path(str(record["phase0_path"])) if record.get("phase0_path") else None
    current_expert_mask_path = Path(str(record["expert_mask_path"]))
    if phase0_path is None or not phase0_path.exists():
        record["expert_roi_status"] = "missing_phase0_image"
    elif not current_expert_mask_path.exists():
        record["expert_roi_status"] = "missing_expert_mask"
    else:
        try:
            record.update(
                expert_roi_record(
                    image_path=phase0_path,
                    mask_path=current_expert_mask_path,
                    margin_mm=roi_margin_mm,
                    strict_affine=strict_affine,
                    affine_atol=affine_atol,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep manifest construction robust.
            record["expert_roi_status"] = "expert_mask_read_or_geometry_error"
            record["expert_roi_error"] = str(exc)

    if pd.isna(row.get("pcr")):
        benchmark_status = "missing_pcr_label"
    elif pd.isna(row.get("split")):
        benchmark_status = "missing_split"
    elif missing_roles:
        benchmark_status = "missing_required_phase"
    elif header_errors:
        benchmark_status = "phase_header_error"
    elif geometry != "matched":
        benchmark_status = f"phase_geometry_{geometry}"
    else:
        benchmark_status = "usable"

    record["benchmark_status"] = benchmark_status
    record["usable_for_whole_volume_benchmark"] = benchmark_status == "usable"
    record["usable_for_multiphase_or_subtraction_benchmark"] = (
        benchmark_status == "usable" and geometry == "matched"
    )
    record["usable_for_expert_roi_benchmark"] = (
        record["usable_for_multiphase_or_subtraction_benchmark"]
        and record["expert_roi_status"] == "expert_mask_roi"
    )

    subtraction_available = bool(record["usable_for_multiphase_or_subtraction_benchmark"])
    record["phase1_minus_phase0_available"] = subtraction_available
    record["phase2_minus_phase0_available"] = subtraction_available
    record["last_phase_minus_phase0_available"] = subtraction_available
    record["whole_volume_input_roles"] = (
        "phase0;phase1;phase2;last_phase;"
        "phase1_minus_phase0;phase2_minus_phase0;last_phase_minus_phase0"
    )
    record["expert_roi_input_roles"] = record["whole_volume_input_roles"]
    return record


def main() -> None:
    args = parse_args()
    mamamia_root = args.mamamia_root.resolve()
    clinical_info = resolve_path(args.clinical_info or mamamia_root / "clinical_and_imaging_info.xlsx")
    splits = resolve_path(args.splits or mamamia_root / "train_test_splits.csv")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical(clinical_info)
    if args.max_rows is not None:
        clinical = clinical.head(args.max_rows).copy()
    split_df = load_splits(splits)
    merged = clinical.merge(split_df, on="patient_id", how="left")

    records = [
        patient_record(
            row=row,
            mamamia_root=mamamia_root,
            roi_margin_mm=args.roi_margin_mm,
            strict_affine=args.strict_affine,
            affine_atol=args.affine_atol,
        )
        for _, row in merged.iterrows()
    ]

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_root / args.manifest_name
    manifest.to_csv(manifest_path, index=False)

    summary = summarize_manifest(
        manifest=manifest,
        args=args,
        clinical_info=clinical_info,
        splits=splits,
        manifest_path=manifest_path,
    )
    summary_path = output_root / args.summary_name
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
