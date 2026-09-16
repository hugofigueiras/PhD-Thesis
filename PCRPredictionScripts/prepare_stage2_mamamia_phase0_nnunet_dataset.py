#!/usr/bin/env python3
"""Prepare a Stage 2 nnU-Net segmentation dataset from MAMA-MIA phase-0 masks.

This creates an nnU-Net raw dataset folder using the official MAMA-MIA train/test
partition. Training labels are placed only for official training patients.
Official test labels are copied/symlinked separately as references for later
held-out evaluation.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent / "outputs" / "stage2_mamamia_phase0_segmentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare nnU-Net raw/preprocessed/results folders for Stage 2 segmentation."
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--clinical-info", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-id", type=int, default=221)
    parser.add_argument("--dataset-name", default="MAMAMIA_Phase0_TumorSeg")
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument(
        "--mask-source",
        choices=("expert", "automatic"),
        default="expert",
        help="Ground-truth mask family to use for segmentation supervision.",
    )
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "hardlink"),
        default="symlink",
        help="How to populate nnU-Net folders.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-cases", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_clinical(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name="dataset_info")
    clinical = clinical.copy()
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    if "pcr" in clinical:
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


def mask_path(mamamia_root: Path, patient_id: str, mask_source: str) -> Path:
    return mamamia_root / "segmentations" / mask_source / f"{patient_id}.nii.gz"


def dataset_folder_name(dataset_id: int, dataset_name: str) -> str:
    return f"Dataset{dataset_id:03d}_{dataset_name}"


def place_file(src: Path, dst: Path, mode: str, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "hardlink":
        dst.hardlink_to(src)
    else:
        shutil.copy2(src, dst)


def split_train_val(train_pool: pd.DataFrame, val_frac: float, seed: int) -> tuple[list[str], list[str]]:
    stratify = None
    if "pcr" in train_pool and train_pool["pcr"].notna().all() and train_pool["pcr"].nunique() == 2:
        stratify = train_pool["pcr"].astype(int)
    train_cases, val_cases = train_test_split(
        train_pool["patient_id"].tolist(),
        test_size=val_frac,
        random_state=seed,
        stratify=stratify,
    )
    return sorted(train_cases), sorted(val_cases)


def command_notes(output_root: Path, folder_name: str, dataset_id: int) -> str:
    raw = output_root / "nnUNet_raw"
    preprocessed = output_root / "nnUNet_preprocessed"
    results = output_root / "nnUNet_results"
    return f"""#!/usr/bin/env bash
set -euo pipefail

export nnUNet_raw="{raw}"
export nnUNet_preprocessed="{preprocessed}"
export nnUNet_results="{results}"

# Plan and preprocess after the dataset has been prepared.
nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity
cp "$nnUNet_raw/{folder_name}/splits_final.json" "$nnUNet_preprocessed/{folder_name}/splits_final.json"

# Keep PCR-test patients out of training. Fold 0 uses the custom splits_final.json
# written by this script under nnUNet_preprocessed/{folder_name}/.
nnUNetv2_train {dataset_id} 3d_fullres 0

# Predict masks for the official test images. The test input remains a whole MRI;
# the predicted mask is generated by the segmentation model.
nnUNetv2_predict \\
  -d {dataset_id} \\
  -i "$nnUNet_raw/{folder_name}/imagesTs" \\
  -o "{output_root / 'predictionsTs'}" \\
  -f 0 \\
  -c 3d_fullres
"""


def main() -> None:
    args = parse_args()
    mamamia_root = args.mamamia_root.resolve()
    clinical_info = resolve_path(args.clinical_info or mamamia_root / "clinical_and_imaging_info.xlsx")
    splits_path = resolve_path(args.splits or mamamia_root / "train_test_splits.csv")
    output_root = args.output_root.resolve()

    folder_name = dataset_folder_name(args.dataset_id, args.dataset_name)
    raw_dataset = output_root / "nnUNet_raw" / folder_name
    preprocessed_dataset = output_root / "nnUNet_preprocessed" / folder_name
    results_root = output_root / "nnUNet_results"
    for subdir in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (raw_dataset / subdir).mkdir(parents=True, exist_ok=True)
    preprocessed_dataset.mkdir(parents=True, exist_ok=True)
    results_root.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical(clinical_info)
    if args.max_cases is not None:
        clinical = clinical.head(args.max_cases).copy()
    split_df = load_splits(splits_path)
    df = clinical.merge(split_df, on="patient_id", how="left")

    records: list[dict[str, object]] = []
    for _, row in df.iterrows():
        patient_id = str(row["patient_id"])
        image = phase_path(mamamia_root, patient_id, args.phase_index)
        mask = mask_path(mamamia_root, patient_id, args.mask_source)
        split = row.get("split")
        usable = bool(image.exists() and mask.exists() and split in {"train", "test"})
        record = row.to_dict()
        record.update(
            {
                "case_id": patient_id,
                "phase_index": int(args.phase_index),
                "image_path": str(image),
                "mask_path": str(mask),
                "mask_source": args.mask_source,
                "segmentation_split": split,
                "usable_for_segmentation": usable,
            }
        )
        records.append(record)
        if not usable:
            continue

        if split == "train":
            place_file(
                image,
                raw_dataset / "imagesTr" / f"{patient_id}_0000.nii.gz",
                args.link_mode,
                args.overwrite,
            )
            place_file(
                mask,
                raw_dataset / "labelsTr" / f"{patient_id}.nii.gz",
                args.link_mode,
                args.overwrite,
            )
        else:
            place_file(
                image,
                raw_dataset / "imagesTs" / f"{patient_id}_0000.nii.gz",
                args.link_mode,
                args.overwrite,
            )
            place_file(
                mask,
                raw_dataset / "labelsTs" / f"{patient_id}.nii.gz",
                args.link_mode,
                args.overwrite,
            )

    manifest = pd.DataFrame.from_records(records)
    usable = manifest[manifest["usable_for_segmentation"]]
    train_pool = usable[usable["segmentation_split"] == "train"].copy()
    test_pool = usable[usable["segmentation_split"] == "test"].copy()
    train_cases, val_cases = split_train_val(train_pool, args.val_frac, args.seed)
    splits_final = [{"train": train_cases, "val": val_cases}]

    dataset_json = {
        "channel_names": {"0": f"MAMA-MIA phase {args.phase_index} DCE-MRI"},
        "labels": {"background": 0, "tumor": 1},
        "numTraining": int(len(train_pool)),
        "file_ending": ".nii.gz",
        "name": args.dataset_name,
        "description": (
            "Stage 2 MAMA-MIA phase-0 tumor segmentation. Official test patients "
            "are held out from labelsTr."
        ),
    }
    (raw_dataset / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, sort_keys=True) + "\n"
    )
    (raw_dataset / "stage2_segmentation_manifest.csv").write_text(
        manifest.to_csv(index=False)
    )
    (preprocessed_dataset / "splits_final.json").write_text(
        json.dumps(splits_final, indent=2, sort_keys=True) + "\n"
    )
    (raw_dataset / "splits_final.json").write_text(
        json.dumps(splits_final, indent=2, sort_keys=True) + "\n"
    )

    summary = {
        "dataset_id": int(args.dataset_id),
        "dataset_name": args.dataset_name,
        "folder_name": folder_name,
        "mamamia_root": str(mamamia_root),
        "clinical_info": str(clinical_info),
        "splits": str(splits_path),
        "raw_dataset": str(raw_dataset),
        "preprocessed_dataset": str(preprocessed_dataset),
        "results_root": str(results_root),
        "phase_index": int(args.phase_index),
        "mask_source": args.mask_source,
        "link_mode": args.link_mode,
        "rows": int(len(manifest)),
        "usable_rows": int(len(usable)),
        "train_labels": int(len(train_pool)),
        "test_images": int(len(test_pool)),
        "custom_train_cases": int(len(train_cases)),
        "custom_val_cases": int(len(val_cases)),
        "usable_split_counts": {
            str(k): int(v)
            for k, v in usable["segmentation_split"].value_counts(dropna=False).items()
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "stage2_segmentation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (output_root / "stage2_nnunet_commands.sh").write_text(
        command_notes(output_root, folder_name, args.dataset_id)
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote raw dataset: {raw_dataset}")
    print(f"Wrote commands:    {output_root / 'stage2_nnunet_commands.sh'}")


if __name__ == "__main__":
    main()
