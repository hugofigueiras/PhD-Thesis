#!/usr/bin/env python3
"""Build a mask-free phase-0 whole-image manifest for pCR classification.

This comparison baseline uses all labeled reconstructed anchor phase-0 images.
It does not require nnU-Net predictions to be non-empty or anatomically plausible;
the whole reconstructed volume is cropped/resized by the classifier dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


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
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "phase0_whole_image"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a mask-free whole-image pCR manifest from phase-0 anchor rows."
    )
    parser.add_argument("--qc-report", type=Path, default=DEFAULT_QC_REPORT)
    parser.add_argument("--clinical-info", type=Path, default=DEFAULT_CLINICAL_INFO)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def parse_triplet(value: object, column_name: str) -> tuple[int, int, int]:
    parts = str(value).split("x")
    if len(parts) != 3:
        raise ValueError(f"Expected {column_name} as XxYxZ, got {value!r}")
    return tuple(int(float(part)) for part in parts)


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
    for _, row in merged.iterrows():
        record = row.to_dict()
        image_path = resolve_path(row["reconstructed_image_path"])
        shape = parse_triplet(row["reconstructed_shape"], "reconstructed_shape")

        record.update(
            {
                "reconstructed_image_path": str(image_path),
                "roi_source": "whole_reconstructed_phase0",
                "roi_status": "whole_image",
                "usable_for_training": bool(image_path.exists() and pd.notna(row.get("pcr"))),
                "image_shape_x": shape[0],
                "image_shape_y": shape[1],
                "image_shape_z": shape[2],
                "center_vox_x": (shape[0] - 1) / 2.0,
                "center_vox_y": (shape[1] - 1) / 2.0,
                "center_vox_z": (shape[2] - 1) / 2.0,
                "crop_start_x": 0,
                "crop_start_y": 0,
                "crop_start_z": 0,
                "crop_end_x": shape[0],
                "crop_end_y": shape[1],
                "crop_end_z": shape[2],
                "crop_clipped_start_x": 0,
                "crop_clipped_start_y": 0,
                "crop_clipped_start_z": 0,
                "crop_clipped_end_x": shape[0],
                "crop_clipped_end_y": shape[1],
                "crop_clipped_end_z": shape[2],
                "crop_needs_padding": False,
            }
        )
        if not image_path.exists():
            record["roi_status"] = "missing_reconstructed_image"
        elif pd.isna(row.get("pcr")):
            record["roi_status"] = "missing_pcr_label"
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_root / "roi_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    usable = manifest[manifest["usable_for_training"]]
    summary = {
        "qc_report": str(args.qc_report.resolve()),
        "clinical_info": str(args.clinical_info.resolve()),
        "splits": str(args.splits.resolve()),
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
