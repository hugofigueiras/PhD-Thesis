from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable

import nibabel as nib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTED_ROOT = PROJECT_ROOT / "Reconstructed_Datasets"
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "ReconstructionScripts" / "qc_reports" / "mamamia_nnunet_qc"
)
DEFAULT_WEIGHTS_ZIP = (
    DEFAULT_MAMAMIA_ROOT
    / "nnUNet_pretrained_weights"
    / "full_image_dce_mri_tumor_segmentation.zip"
)

DATASET_TO_RECON_DIR = {
    "ISPY1": "ISPY1-Volumes",
    "ISPY2": "ISPY2-Volumes",
    "NACT": "NACT-Volumes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage MAMA-MIA original phases and reconstructed phases as one-channel "
            "nnU-Net inference cases, then write a manifest for downstream QC."
        )
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--reconstructed-root", type=Path, default=DEFAULT_RECONSTRUCTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--weights-zip", type=Path, default=DEFAULT_WEIGHTS_ZIP)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASET_TO_RECON_DIR),
        choices=sorted(DATASET_TO_RECON_DIR),
        help="Datasets to stage from the reconstructed dataset root.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help=(
            "Optional MAMA-MIA patient IDs to stage, for example ISPY1_1208 "
            "or ISPY2_145513."
        ),
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Limit staged patients after filtering. Useful for smoke checks.",
    )
    parser.add_argument(
        "--max-phases-per-exam",
        type=int,
        default=None,
        help="Limit phases staged per patient/exam. Useful for smoke checks.",
    )
    parser.add_argument(
        "--phase-indices",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Only stage these phase indices, for example '--phase-indices 0' "
            "for a fast phase0-only QC run."
        ),
    )
    parser.add_argument(
        "--timepoints",
        choices=("all", "anchor"),
        default="all",
        help="Stage all reconstructed dates or only the MAMA-MIA anchor date.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="How to create nnU-Net input files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing staged files.",
    )
    parser.add_argument(
        "--include-quarantine",
        action="store_true",
        help="Also stage quarantined ISPY1 geometry cases.",
    )
    parser.add_argument(
        "--predict-mamamia",
        action="store_true",
        help=(
            "Also include nnU-Net inference for the original MAMA-MIA phases. "
            "By default only reconstructed phases are predicted."
        ),
    )
    return parser.parse_args()


def normalized_patient_id(patient_id: str, dataset: str) -> str:
    if dataset == "ISPY2":
        return patient_id.replace("_", "-")
    return patient_id


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def parse_phase_index(path: Path) -> int:
    match = re.search(r"(?:_|phase)(\d+)\.nii\.gz$", path.name)
    if not match:
        raise ValueError(f"Could not parse phase index from {path}")
    return int(match.group(1))


def date_to_reconstructed_folder(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text.replace("/", "-")
    return parsed.strftime("%m-%d-%Y")


def iter_selected(
    paths: Iterable[Path],
    limit: int | None,
    phase_indices: set[int] | None,
) -> list[Path]:
    items = sorted(paths, key=lambda item: (parse_phase_index(item), item.name))
    if phase_indices is not None:
        items = [item for item in items if parse_phase_index(item) in phase_indices]
    if limit is not None:
        return items[:limit]
    return items


def image_header_summary(path: Path) -> dict[str, str]:
    img = nib.load(str(path))
    zooms = img.header.get_zooms()[: len(img.shape)]
    return {
        "shape": "x".join(str(dim) for dim in img.shape),
        "spacing": "x".join(f"{zoom:.6g}" for zoom in zooms),
    }


def create_input_file(source: Path, destination: Path, link_mode: str, overwrite: bool) -> str:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return "exists"
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_mode == "symlink":
        os.symlink(source.resolve(), destination)
        return "symlinked"

    shutil.copy2(source, destination)
    return "copied"


def load_clinical_info(mamamia_root: Path, datasets: set[str]) -> pd.DataFrame:
    clinical_path = mamamia_root / "clinical_and_imaging_info.xlsx"
    df = pd.read_excel(clinical_path)
    df = df[df["dataset"].isin(datasets)].copy()
    df["patient_id"] = df["patient_id"].astype(str)
    df["anchor_reconstructed_date"] = df["acquisition_date"].map(date_to_reconstructed_folder)
    df = df.sort_values(["dataset", "patient_id"], kind="stable").reset_index(drop=True)
    return df


def stage_mamamia_phases(
    row: pd.Series,
    mamamia_root: Path,
    input_dir: Path,
    link_mode: str,
    overwrite: bool,
    max_phases: int | None,
    phase_indices: set[int] | None,
) -> list[dict[str, object]]:
    patient_id = row["patient_id"]
    image_dir = mamamia_root / "images" / patient_id
    expert_mask = mamamia_root / "segmentations" / "expert" / f"{patient_id}.nii.gz"
    automatic_mask = mamamia_root / "segmentations" / "automatic" / f"{patient_id}.nii.gz"
    phase_paths = iter_selected(
        image_dir.glob(f"{patient_id}_*.nii.gz"), max_phases, phase_indices
    )

    rows: list[dict[str, object]] = []
    for source in phase_paths:
        phase_index = parse_phase_index(source)
        case_id = f"mamamia__{safe_token(patient_id)}__phase{phase_index:04d}"
        staged = input_dir / f"{case_id}_0000.nii.gz"
        link_status = create_input_file(source, staged, link_mode, overwrite)
        header = image_header_summary(source)
        rows.append(
            {
                "collection": "mamamia",
                "case_id": case_id,
                "staged_image_path": str(staged),
                "source_image_path": str(source),
                "patient_id": patient_id,
                "reconstructed_patient_id": normalized_patient_id(patient_id, row["dataset"]),
                "dataset": row["dataset"],
                "reconstructed_dataset_dir": DATASET_TO_RECON_DIR.get(row["dataset"], ""),
                "exam_date": "",
                "anchor_reconstructed_date": row["anchor_reconstructed_date"],
                "is_anchor_date": "",
                "phase_index": phase_index,
                "mamamia_num_phases": row.get("num_phases", ""),
                "paired_mamamia_case_id": case_id,
                "paired_mamamia_image_path": str(source),
                "expert_mask_path": str(expert_mask) if expert_mask.exists() else "",
                "automatic_mask_path": str(automatic_mask) if automatic_mask.exists() else "",
                "source_shape": header["shape"],
                "source_spacing": header["spacing"],
                "stage_status": link_status,
            }
        )
    return rows


def reconstructed_patient_roots(
    reconstructed_root: Path,
    row: pd.Series,
    include_quarantine: bool,
) -> list[tuple[Path, str]]:
    dataset = row["dataset"]
    recon_dir_name = DATASET_TO_RECON_DIR[dataset]
    recon_patient_id = normalized_patient_id(row["patient_id"], dataset)
    roots = [(reconstructed_root / recon_dir_name / recon_patient_id, recon_dir_name)]

    if include_quarantine and dataset == "ISPY1":
        quarantine = (
            reconstructed_root
            / "Quarantine"
            / "ISPY1-Geometry-Flagged-Patients"
            / recon_dir_name
            / recon_patient_id
        )
        roots.append((quarantine, f"Quarantine/{recon_dir_name}"))
    return roots


def stage_reconstructed_phases(
    row: pd.Series,
    mamamia_root: Path,
    reconstructed_root: Path,
    input_dir: Path,
    link_mode: str,
    overwrite: bool,
    max_phases: int | None,
    phase_indices: set[int] | None,
    timepoints: str,
    include_quarantine: bool,
) -> list[dict[str, object]]:
    patient_id = row["patient_id"]
    expert_mask = mamamia_root / "segmentations" / "expert" / f"{patient_id}.nii.gz"
    automatic_mask = mamamia_root / "segmentations" / "automatic" / f"{patient_id}.nii.gz"
    anchor_date = str(row["anchor_reconstructed_date"])

    rows: list[dict[str, object]] = []
    for patient_root, recon_dir_name in reconstructed_patient_roots(
        reconstructed_root, row, include_quarantine
    ):
        if not patient_root.exists():
            continue

        exam_dirs = [path for path in patient_root.iterdir() if path.is_dir()]
        exam_dirs = sorted(exam_dirs, key=lambda item: item.name)
        if timepoints == "anchor":
            exam_dirs = [path for path in exam_dirs if path.name == anchor_date]

        for exam_dir in exam_dirs:
            is_anchor = exam_dir.name == anchor_date
            phase_paths = iter_selected(
                exam_dir.glob("volume_phase*.nii.gz"), max_phases, phase_indices
            )
            for source in phase_paths:
                phase_index = parse_phase_index(source)
                date_token = safe_token(exam_dir.name.replace("-", ""))
                case_id = (
                    f"recon__{safe_token(patient_id)}__{date_token}"
                    f"__phase{phase_index:04d}"
                )
                staged = input_dir / f"{case_id}_0000.nii.gz"
                link_status = create_input_file(source, staged, link_mode, overwrite)
                header = image_header_summary(source)

                paired_mamamia_image = (
                    mamamia_root
                    / "images"
                    / patient_id
                    / f"{patient_id}_{phase_index:04d}.nii.gz"
                )
                paired_case = f"mamamia__{safe_token(patient_id)}__phase{phase_index:04d}"
                rows.append(
                    {
                        "collection": "reconstructed",
                        "case_id": case_id,
                        "staged_image_path": str(staged),
                        "source_image_path": str(source),
                        "patient_id": patient_id,
                        "reconstructed_patient_id": normalized_patient_id(
                            patient_id, row["dataset"]
                        ),
                        "dataset": row["dataset"],
                        "reconstructed_dataset_dir": recon_dir_name,
                        "exam_date": exam_dir.name,
                        "anchor_reconstructed_date": anchor_date,
                        "is_anchor_date": str(is_anchor),
                        "phase_index": phase_index,
                        "mamamia_num_phases": row.get("num_phases", ""),
                        "paired_mamamia_case_id": paired_case,
                        "paired_mamamia_image_path": (
                            str(paired_mamamia_image)
                            if paired_mamamia_image.exists()
                            else ""
                        ),
                        "expert_mask_path": str(expert_mask) if expert_mask.exists() else "",
                        "automatic_mask_path": (
                            str(automatic_mask) if automatic_mask.exists() else ""
                        ),
                        "source_shape": header["shape"],
                        "source_spacing": header["spacing"],
                        "stage_status": link_status,
                    }
                )
    return rows


def write_command_notes(output_root: Path, weights_zip: Path, predict_mamamia: bool) -> None:
    commands_path = output_root / "nnunet_command_notes.sh"
    input_root = output_root / "nnunet_inputs"
    prediction_root = output_root / "nnunet_predictions"
    results_root = output_root / "nnUNet_results"
    model_root = (
        results_root
        / "Dataset105_full_image"
        / "nnUNetTrainer__nnUNetPlans__3d_fullres"
    )
    tmp_weights = output_root / "_weights_unpacked"

    mamamia_prediction_block = ""
    if predict_mamamia:
        mamamia_prediction_block = f"""
nnUNetv2_predict \\
  -i "{input_root / 'mamamia_phases'}" \\
  -o "{prediction_root / 'mamamia_phases'}" \\
  -d 105 \\
  -c 3d_fullres \\
  -f 0 1 2 3 4 \\
  -chk checkpoint_final.pth
"""

    text = f"""#!/usr/bin/env bash
set -euo pipefail

# Install nnU-Net v2 in the project venv if needed:
# ./.venv/bin/pip install nnunetv2

# Unpack the MAMA-MIA weights into the layout expected by nnUNetv2_predict.
mkdir -p "{model_root}"
mkdir -p "{tmp_weights}"
unzip -n "{weights_zip}" -d "{tmp_weights}"
cp -R "{tmp_weights}/full_image_dce_mri_tumor_segmentation/." "{model_root}/"

export nnUNet_results="{results_root}"
export nnUNet_raw="{output_root / 'nnUNet_raw'}"
export nnUNet_preprocessed="{output_root / 'nnUNet_preprocessed'}"
{mamamia_prediction_block}

nnUNetv2_predict \\
  -i "{input_root / 'reconstructed_phases'}" \\
  -o "{prediction_root / 'reconstructed_phases'}" \\
  -d 105 \\
  -c 3d_fullres \\
  -f 0 1 2 3 4 \\
  -chk checkpoint_final.pth
"""
    commands_path.write_text(text)


def main() -> None:
    args = parse_args()
    mamamia_root = args.mamamia_root.resolve()
    reconstructed_root = args.reconstructed_root.resolve()
    output_root = args.output_root.resolve()
    input_root = output_root / "nnunet_inputs"
    mamamia_input_dir = input_root / "mamamia_phases"
    reconstructed_input_dir = input_root / "reconstructed_phases"
    output_root.mkdir(parents=True, exist_ok=True)
    phase_indices = set(args.phase_indices) if args.phase_indices is not None else None

    datasets = set(args.datasets)
    clinical_df = load_clinical_info(mamamia_root, datasets)
    if args.patients:
        wanted = {str(patient) for patient in args.patients}
        clinical_df = clinical_df[clinical_df["patient_id"].isin(wanted)].copy()
    if args.max_patients is not None:
        clinical_df = clinical_df.head(args.max_patients).copy()

    manifest_rows: list[dict[str, object]] = []
    for _, row in clinical_df.iterrows():
        if row["dataset"] not in DATASET_TO_RECON_DIR:
            continue
        manifest_rows.extend(
            stage_mamamia_phases(
                row=row,
                mamamia_root=mamamia_root,
                input_dir=mamamia_input_dir,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
                max_phases=args.max_phases_per_exam,
                phase_indices=phase_indices,
            )
        )
        manifest_rows.extend(
            stage_reconstructed_phases(
                row=row,
                mamamia_root=mamamia_root,
                reconstructed_root=reconstructed_root,
                input_dir=reconstructed_input_dir,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
                max_phases=args.max_phases_per_exam,
                phase_indices=phase_indices,
                timepoints=args.timepoints,
                include_quarantine=args.include_quarantine,
            )
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = output_root / "qc_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    write_command_notes(output_root, args.weights_zip.resolve(), args.predict_mamamia)

    print(f"Patients considered: {len(clinical_df)}")
    print(f"Manifest rows: {len(manifest_df)}")
    if not manifest_df.empty:
        print(manifest_df["collection"].value_counts().to_string())
        print(manifest_df["stage_status"].value_counts().to_string())
    print(f"Manifest: {manifest_path}")
    print(f"nnU-Net command notes: {output_root / 'nnunet_command_notes.sh'}")


if __name__ == "__main__":
    main()
