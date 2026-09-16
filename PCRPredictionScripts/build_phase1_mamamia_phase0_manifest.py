#!/usr/bin/env python3
"""Build an original MAMA-MIA phase-0 whole-image manifest for pCR classification.

This manifest is compatible with train_phase0_nnunet_pcr_classifier.py. It uses
the original MAMA-MIA NIfTI images, keeps only phase index 0 by default, and
records full-volume crop coordinates so resizing can happen on the fly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "phase1_mamamia_phase0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a pCR classification manifest from original MAMA-MIA phase-0 "
            "NIfTI images."
        )
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--clinical-info", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--phase-index",
        type=int,
        default=0,
        help="MAMA-MIA phase index to use. Phase 0 maps to *_0000.nii.gz.",
    )
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


def phase_path(mamamia_root: Path, patient_id: str, phase_index: int) -> Path:
    return mamamia_root / "images" / patient_id / f"{patient_id}_{phase_index:04d}.nii.gz"


def count_available_phases(mamamia_root: Path, patient_id: str) -> int:
    image_dir = mamamia_root / "images" / patient_id
    if not image_dir.exists():
        return 0
    return len(sorted(image_dir.glob(f"{patient_id}_*.nii.gz")))


def image_record(image_path: Path) -> dict[str, object]:
    image = nib.load(str(image_path))
    spacing = image.header.get_zooms()[:3]
    return {
        "image_shape_x": int(image.shape[0]),
        "image_shape_y": int(image.shape[1]),
        "image_shape_z": int(image.shape[2]),
        "image_spacing_x": float(spacing[0]),
        "image_spacing_y": float(spacing[1]),
        "image_spacing_z": float(spacing[2]),
        "image_dtype": str(image.get_data_dtype()),
        "center_vox_x": (int(image.shape[0]) - 1) / 2.0,
        "center_vox_y": (int(image.shape[1]) - 1) / 2.0,
        "center_vox_z": (int(image.shape[2]) - 1) / 2.0,
        "crop_start_x": 0,
        "crop_start_y": 0,
        "crop_start_z": 0,
        "crop_end_x": int(image.shape[0]),
        "crop_end_y": int(image.shape[1]),
        "crop_end_z": int(image.shape[2]),
        "crop_clipped_start_x": 0,
        "crop_clipped_start_y": 0,
        "crop_clipped_start_z": 0,
        "crop_clipped_end_x": int(image.shape[0]),
        "crop_clipped_end_y": int(image.shape[1]),
        "crop_clipped_end_z": int(image.shape[2]),
        "crop_needs_padding": False,
    }


def summarize_manifest(
    manifest: pd.DataFrame,
    args: argparse.Namespace,
    clinical_info: Path,
    splits: Path,
) -> dict[str, object]:
    usable = manifest[manifest["usable_for_training"]]
    z_values = manifest["image_shape_z"].dropna().astype(int)
    return {
        "mamamia_root": str(args.mamamia_root.resolve()),
        "clinical_info": str(clinical_info.resolve()),
        "splits": str(splits.resolve()),
        "phase_index": int(args.phase_index),
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
        "image_shape_x_min": int(manifest["image_shape_x"].dropna().min()),
        "image_shape_x_max": int(manifest["image_shape_x"].dropna().max()),
        "image_shape_y_min": int(manifest["image_shape_y"].dropna().min()),
        "image_shape_y_max": int(manifest["image_shape_y"].dropna().max()),
        "image_shape_z_min": int(z_values.min()),
        "image_shape_z_max": int(z_values.max()),
        "image_shape_z_median": float(z_values.median()),
        "preprocessing": {
            "disk_preprocessing": "none",
            "training_time_operations": [
                "load original NIfTI phase image",
                "full-volume crop",
                "0.5/99.5 percentile clipping",
                "per-volume z-score normalization",
                "aspect-preserving trilinear resize plus zero padding",
            ],
            "recommended_input_shape": [256, 256, 64],
            "recommended_resize_mode": "pad",
        },
    }


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

    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        patient_id = str(row["patient_id"])
        image_path = phase_path(mamamia_root, patient_id, args.phase_index)
        expert_mask_path = mamamia_root / "segmentations" / "expert" / f"{patient_id}.nii.gz"
        automatic_mask_path = mamamia_root / "segmentations" / "automatic" / f"{patient_id}.nii.gz"

        record = row.to_dict()
        record.update(
            {
                "phase_index": int(args.phase_index),
                "case_id": f"mamamia__{patient_id}__phase{args.phase_index:04d}",
                "image_path": str(image_path),
                "reconstructed_image_path": str(image_path),
                "expert_mask_path": str(expert_mask_path),
                "automatic_mask_path": str(automatic_mask_path),
                "expert_mask_exists": expert_mask_path.exists(),
                "automatic_mask_exists": automatic_mask_path.exists(),
                "num_available_phases": count_available_phases(mamamia_root, patient_id),
                "roi_source": "original_mamamia_whole_phase",
                "roi_status": "mamamia_phase0_whole_image",
            }
        )

        if not image_path.exists():
            record["roi_status"] = "missing_phase_image"
        elif pd.isna(row.get("pcr")):
            record["roi_status"] = "missing_pcr_label"
        elif pd.isna(row.get("split")):
            record["roi_status"] = "missing_split"
        else:
            record.update(image_record(image_path))

        record["usable_for_training"] = record["roi_status"] == "mamamia_phase0_whole_image"
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = summarize_manifest(manifest, args, clinical_info, splits)
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
