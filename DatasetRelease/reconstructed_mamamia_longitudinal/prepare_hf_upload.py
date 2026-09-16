#!/usr/bin/env python3
"""Prepare and optionally upload the public Hugging Face dataset folder.

The staging folder intentionally excludes quarantine/internal audit files. Image
files are hardlinked by default, so staging does not duplicate the NIfTI data.
Deleting the staging folder does not delete the original reconstructed volumes.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, get_token


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RELEASE_ROOT = PROJECT_ROOT / "DatasetRelease" / "reconstructed_mamamia_longitudinal"
METADATA_ROOT = RELEASE_ROOT / "metadata"
DEFAULT_STAGE_ROOT = RELEASE_ROOT / "hf_upload_staging"

PUBLIC_METADATA_FILES = {
    "clinical_and_imaging_info_active_reconstructed.csv": (
        "MAMA-MIA clinical/imaging table filtered to active reconstructed patients."
    ),
    "patient_manifest.csv": "One row per active reconstructed patient.",
    "exam_manifest.csv": "One row per active reconstructed exam/timepoint.",
    "phase_manifest.csv": "One row per active NIfTI phase file.",
}

PUBLIC_METADATA_COLUMNS_TO_DROP = {
    "clinical_and_imaging_info_active_reconstructed.csv": [
        "reconstructed_release_status",
        "include_in_reconstructed_release",
        "quarantine_cases",
        "quarantine_flagged_exam_dates",
        "quarantine_issues",
    ],
    "exam_manifest.csv": [
        "quarantined_exam_exists",
    ],
}

PUBLIC_CLINICAL_COLUMN_GROUPS = {
    "Identifiers/source provenance": [
        "patient_id",
        "dataset",
    ],
    "Clinical, treatment, outcome, and tumor biology": [
        "bilateral_breast_cancer",
        "multifocal_cancer",
        "nac_agent",
        "endocrine_therapy",
        "anti_her2_neu_therapy",
        "pcr",
        "mastectomy_post_nac",
        "days_to_follow_up",
        "days_to_recurrence",
        "days_to_metastasis",
        "days_to_death",
        "hr",
        "er",
        "pr",
        "her2",
        "mammaprint",
        "oncotype_score",
        "nottingham_grade",
        "tumor_subtype",
    ],
    "Demographics and body habitus": [
        "age",
        "menopause",
        "ethnicity",
        "has_implant",
        "weight",
        "patient_size",
        "bmi_group",
        "breast_density",
    ],
    "Imaging, acquisition, scanner, and TCIA series metadata": [
        "view",
        "bilateral_mri",
        "num_phases",
        "fat_suppressed",
        "field_strength",
        "image_rows",
        "image_columns",
        "num_slices",
        "pixel_spacing",
        "slice_thickness",
        "site",
        "manufacturer",
        "scanner_model",
        "high_bit",
        "window_center",
        "window_width",
        "echo_time",
        "repetition_time",
        "acquisition_times",
        "acquisition_date",
        "tcia_series_uid",
    ],
    "Reconstruction and longitudinal linkage": [
        "active_reconstructed_patient_id",
        "active_volume_root",
        "active_patient_rel_path",
        "num_reconstructed_exams",
        "num_reconstructed_extra_exams",
        "num_reconstructed_phase_files",
        "reconstructed_exam_dates",
        "reconstructed_exam_dates_iso",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stage active reconstructed data for Hugging Face and optionally upload it."
        )
    )
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "symlink", "copy"),
        default="hardlink",
        help="How to place NIfTI files in the staging folder.",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Delete the existing staging folder before rebuilding it.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned file counts and size; do not stage or upload.",
    )
    parser.add_argument("--repo-id", help="Hugging Face dataset repo id, e.g. user/name.")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the staged folder to the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch/revision to upload to. Defaults to the repo default branch.",
    )
    return parser.parse_args()


def relative_to_project(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def safe_replace_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def write_public_metadata(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    drop_columns = PUBLIC_METADATA_COLUMNS_TO_DROP.get(src.name, [])
    if not drop_columns:
        safe_replace_file(src, dst)
        return

    df = pd.read_csv(src)
    existing_drop_columns = [column for column in drop_columns if column in df.columns]
    df = df.drop(columns=existing_drop_columns)
    df.to_csv(dst, index=False)


def validate_public_clinical_column_groups() -> tuple[int, dict[str, list[str]]]:
    metadata_name = "clinical_and_imaging_info_active_reconstructed.csv"
    src = METADATA_ROOT / metadata_name
    source_columns = list(pd.read_csv(src, nrows=0).columns)
    drop_columns = set(PUBLIC_METADATA_COLUMNS_TO_DROP.get(metadata_name, []))
    public_columns = [column for column in source_columns if column not in drop_columns]

    grouped_columns = [
        column
        for columns in PUBLIC_CLINICAL_COLUMN_GROUPS.values()
        for column in columns
    ]
    duplicate_columns = sorted(
        {column for column in grouped_columns if grouped_columns.count(column) > 1}
    )
    missing_columns = [column for column in public_columns if column not in grouped_columns]
    extra_columns = [column for column in grouped_columns if column not in public_columns]

    if duplicate_columns or missing_columns or extra_columns:
        message_parts = []
        if duplicate_columns:
            message_parts.append(f"duplicate grouped columns: {duplicate_columns}")
        if missing_columns:
            message_parts.append(f"ungrouped public columns: {missing_columns}")
        if extra_columns:
            message_parts.append(f"grouped columns absent from public CSV: {extra_columns}")
        raise ValueError("; ".join(message_parts))

    return len(public_columns), PUBLIC_CLINICAL_COLUMN_GROUPS


def format_count_distribution(distribution: dict[int, int], singular: str, plural: str) -> str:
    parts = []
    for value, count in distribution.items():
        noun = singular if value == 1 else plural
        parts.append(f"{count} patients have {value} {noun}")
    if len(parts) == 1:
        return parts[0]
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def format_column_group_table(column_groups: dict[str, list[str]]) -> str:
    rows = ["| Column group | Count | Columns |", "|---|---:|---|"]
    for group, columns in column_groups.items():
        column_text = ", ".join(f"`{column}`" for column in columns)
        rows.append(f"| {group} | {len(columns)} | {column_text} |")
    return "\n".join(rows)


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        try:
            if os.path.samefile(src, dst):
                return
        except OSError:
            pass
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(os.path.relpath(src, dst.parent))
    else:
        try:
            os.link(src, dst)
        except OSError:
            # Cross-device hardlinks fail; symlink keeps staging lightweight.
            dst.symlink_to(os.path.relpath(src, dst.parent))


def write_dataset_card(
    stage_root: Path,
    patient_count: int,
    exam_count: int,
    phase_count: int,
    dataset_counts: dict[str, dict[str, int]],
    clinical_column_count: int,
    clinical_column_groups: dict[str, list[str]],
    exam_count_distribution: dict[int, int],
    min_phase_count: int,
    max_phase_count: int,
) -> None:
    clinical_table = format_column_group_table(clinical_column_groups)
    exam_distribution_text = format_count_distribution(
        exam_count_distribution, singular="exam/timepoint", plural="exams/timepoints"
    )
    text = f"""---
license: cc-by-nc-4.0
pretty_name: Reconstructed MAMA-MIA Longitudinal DCE-MRI
tags:
- medical-imaging
- mri
- breast-cancer
- dce-mri
- longitudinal
- pathology-complete-response
size_categories:
- 1K<n<10K
---

# Reconstructed MAMA-MIA Longitudinal DCE-MRI

This dataset contains reconstructed longitudinal breast DCE-MRI volumes and
patient-level tabular metadata for the MAMA-MIA patients whose source data came
from ISPY1, ISPY2, and Breast MRI NACT Pilot. Patients excluded during local QC
are not included in this public upload.

## Dataset Description

This is a multimodal, longitudinal breast cancer response dataset. The imaging
modality is volumetric DCE-MRI stored as NIfTI phase files
(`volume_phase*.nii.gz`). The tabular modality contains patient-level clinical,
demographic/body-habitus, treatment, tumor biology, outcome, scanner, and
acquisition metadata inherited from MAMA-MIA, plus reconstruction fields added
for this longitudinal release.

The main response label inherited from MAMA-MIA is `pcr`, pathological complete
response to neoadjuvant chemotherapy. Longitudinal structure is represented by
multiple reconstructed MRI exams/timepoints per patient when the source TCIA
data contained additional exams. Each patient folder contains one or more
exam-date folders, and each exam folder contains the available DCE phase volumes
for that timepoint.

## Contents

- Active patients: {patient_count}
- Active exams/timepoints: {exam_count}
- Active NIfTI phase files: {phase_count}
- Source cohorts: ISPY1, ISPY2, and Breast MRI NACT Pilot
- Timepoints per patient: {exam_distribution_text}
- DCE phase files per exam: {min_phase_count}-{max_phase_count}

## Clinical And Imaging Table

`metadata/clinical_and_imaging_info_active_reconstructed.csv` contains one row
per active released patient. It has {clinical_column_count} public columns: the
50 MAMA-MIA clinical/imaging fields for the active reconstructed cohort plus 8
reconstruction/linkage fields added for this longitudinal release.

Using the grouping below, the public table includes 19 clinical/treatment/
outcome/tumor-biology variables, 8 demographic/body-habitus variables, 21
imaging/acquisition/scanner/TCIA-series variables, 8 reconstruction/linkage
variables, and 2 identifier/source-provenance variables.

{clinical_table}

## Dataset Organization

```text
.
|-- README.md
|-- metadata/
|   |-- clinical_and_imaging_info_active_reconstructed.csv
|   |-- patient_manifest.csv
|   |-- exam_manifest.csv
|   |-- phase_manifest.csv
|   `-- data_dictionary.csv
`-- Reconstructed_Datasets/
    |-- ISPY1-Volumes/                         # {dataset_counts["ISPY1"]["patients"]} patients, {dataset_counts["ISPY1"]["exams"]} exams, {dataset_counts["ISPY1"]["phases"]} phase files
    |   `-- ISPY1_1001/
    |       `-- 10-13-1984/
    |           |-- volume_phase0.nii.gz
    |           |-- volume_phase1.nii.gz
    |           `-- volume_phase2.nii.gz
    |-- ISPY2-Volumes/                         # {dataset_counts["ISPY2"]["patients"]} patients, {dataset_counts["ISPY2"]["exams"]} exams, {dataset_counts["ISPY2"]["phases"]} phase files
    |   `-- ISPY2-100899/
    |       `-- 10-26-2002/
    |           |-- volume_phase0.nii.gz
    |           |-- volume_phase1.nii.gz
    |           `-- ...
    `-- NACT-Volumes/                          # {dataset_counts["NACT"]["patients"]} patients, {dataset_counts["NACT"]["exams"]} exams, {dataset_counts["NACT"]["phases"]} phase files
        `-- NACT_01/
            `-- <MM-DD-YYYY>/
                |-- volume_phase0.nii.gz
                |-- volume_phase1.nii.gz
                `-- ...
```

Public-facing `patient_id` values in the metadata follow MAMA-MIA naming. ISPY2
folders use hyphens locally, for example `ISPY2-100899`; the metadata also
records the MAMA-MIA-style identifier `ISPY2_100899`.

## Metadata

- `metadata/clinical_and_imaging_info_active_reconstructed.csv`
- `metadata/patient_manifest.csv`
- `metadata/exam_manifest.csv`
- `metadata/phase_manifest.csv`
- `metadata/data_dictionary.csv`

The phase manifest contains project-relative paths to every uploaded NIfTI file.

## Sources And Citations

This release is derived from public TCIA/MAMA-MIA source data. Users should cite
MAMA-MIA, the original source datasets represented here, and TCIA.

### MAMA-MIA

```bibtex
@article{{garrucho2025,
  title={{A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations}},
  author={{Garrucho, Lidia and others}},
  journal={{Scientific Data}},
  year={{2025}},
  doi={{10.1038/s41597-025-04707-4}},
  pages={{453}},
  number={{1}},
  volume={{12}}
}}
```

### Original Source Datasets

```bibtex
@dataset{{newitt2016ispy1,
  title={{Multi-center breast DCE-MRI data and segmentations from patients in the I-SPY 1/ACRIN 6657 trials}},
  author={{Newitt, David and Hylton, Nola and I-SPY 1 Network and ACRIN 6657 Trial Team}},
  year={{2016}},
  publisher={{The Cancer Imaging Archive}},
  doi={{10.7937/K9/TCIA.2016.HdHpgJLK}}
}}

@dataset{{newitt2016nact,
  title={{Single site breast DCE-MRI data and segmentations from patients undergoing neoadjuvant chemotherapy}},
  author={{Newitt, David and Hylton, Nola}},
  year={{2016}},
  version={{3}},
  publisher={{The Cancer Imaging Archive}},
  doi={{10.7937/K9/TCIA.2016.QHsyhJKy}}
}}

@dataset{{li2022ispy2,
  title={{I-SPY 2 Breast Dynamic Contrast Enhanced MRI Trial (ISPY2)}},
  author={{Li, W. and Newitt, D. C. and Gibbs, J. and Wilmes, L. J. and Jones, E. F. and Arasu, V. A. and Strand, F. and Onishi, N. and Nguyen, A. A.-T. and Kornak, J. and Joe, B. N. and Price, E. R. and Ojeda-Fournier, H. and Eghtedari, M. and Zamora, K. W. and Woodard, S. A. and Umphrey, H. and Bernreuter, W. and Nelson, M. and Hylton, N. M.}},
  year={{2022}},
  version={{1}},
  publisher={{The Cancer Imaging Archive}},
  doi={{10.7937/TCIA.D8Z0-9T85}}
}}
```

### TCIA

```bibtex
@article{{clark2013tcia,
  title={{The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository}},
  author={{Clark, K. and Vendt, B. and Smith, K. and Freymann, J. and Kirby, J. and Koppel, P. and Moore, S. and Phillips, S. and Maffitt, D. and Pringle, M. and Tarbox, L. and Prior, F.}},
  journal={{Journal of Digital Imaging}},
  volume={{26}},
  number={{6}},
  pages={{1045--1057}},
  year={{2013}},
  doi={{10.1007/s10278-013-9622-7}}
}}
```

## Use Restrictions

Do not attempt to identify or contact individual participants. Follow TCIA and
MAMA-MIA data usage policies and attribution requirements. This dataset is shared
under a non-commercial Creative Commons license to match the MAMA-MIA license.
"""
    (stage_root / "README.md").write_text(text)


def write_public_data_dictionary(stage_root: Path) -> None:
    rows = [
        {"file": name, "description": description}
        for name, description in PUBLIC_METADATA_FILES.items()
    ]
    pd.DataFrame.from_records(rows).to_csv(
        stage_root / "metadata" / "data_dictionary.csv", index=False
    )


def write_gitattributes(stage_root: Path) -> None:
    (stage_root / ".gitattributes").write_text("*.nii.gz filter=lfs diff=lfs merge=lfs -text\n")


def summarize_inputs() -> tuple[
    pd.DataFrame,
    int,
    int,
    int,
    dict[str, dict[str, int]],
    dict[int, int],
    int,
    int,
]:
    phase_manifest = pd.read_csv(METADATA_ROOT / "phase_manifest.csv")
    patient_manifest = pd.read_csv(METADATA_ROOT / "patient_manifest.csv")
    exam_manifest = pd.read_csv(METADATA_ROOT / "exam_manifest.csv")
    total_bytes = int(phase_manifest["file_size_bytes"].sum())
    dataset_counts: dict[str, dict[str, int]] = {}
    for dataset in ("ISPY1", "ISPY2", "NACT"):
        dataset_counts[dataset] = {
            "patients": int((patient_manifest["dataset"] == dataset).sum()),
            "exams": int((exam_manifest["dataset"] == dataset).sum()),
            "phases": int((phase_manifest["dataset"] == dataset).sum()),
        }
    exam_count_distribution = {
        int(exams): int(patients)
        for exams, patients in patient_manifest["num_exams"]
        .value_counts()
        .sort_index()
        .items()
    }
    min_phase_count = int(exam_manifest["phase_count"].min())
    max_phase_count = int(exam_manifest["phase_count"].max())
    return (
        phase_manifest,
        len(patient_manifest),
        len(exam_manifest),
        total_bytes,
        dataset_counts,
        exam_count_distribution,
        min_phase_count,
        max_phase_count,
    )


def stage_dataset(stage_root: Path, link_mode: str, clear: bool, dry_run: bool) -> None:
    (
        phase_manifest,
        patient_count,
        exam_count,
        total_bytes,
        dataset_counts,
        exam_count_distribution,
        min_phase_count,
        max_phase_count,
    ) = summarize_inputs()
    clinical_column_count, clinical_column_groups = validate_public_clinical_column_groups()
    phase_count = len(phase_manifest)

    gib = total_bytes / 1024**3
    print(
        f"Active upload plan: {patient_count} patients, {exam_count} exams, "
        f"{phase_count} NIfTI files, {gib:.1f} GiB"
    )
    print(f"Stage root: {relative_to_project(stage_root)}")
    print("Excluded: quarantine folders and internal nonreleased/status metadata")

    if dry_run:
        return

    if clear and stage_root.exists():
        shutil.rmtree(stage_root)
    (stage_root / "metadata").mkdir(parents=True, exist_ok=True)

    write_dataset_card(
        stage_root,
        patient_count,
        exam_count,
        phase_count,
        dataset_counts,
        clinical_column_count,
        clinical_column_groups,
        exam_count_distribution,
        min_phase_count,
        max_phase_count,
    )
    write_public_data_dictionary(stage_root)
    write_gitattributes(stage_root)

    for filename in PUBLIC_METADATA_FILES:
        write_public_metadata(METADATA_ROOT / filename, stage_root / "metadata" / filename)

    for row in phase_manifest.itertuples(index=False):
        rel_path = Path(str(row.phase_rel_path))
        src = PROJECT_ROOT / rel_path
        dst = stage_root / rel_path
        if not src.exists():
            raise FileNotFoundError(src)
        place_file(src, dst, link_mode)

    print("Staging complete.")


def upload_dataset(stage_root: Path, repo_id: str, revision: str | None) -> None:
    if not repo_id:
        raise SystemExit("--repo-id is required with --upload")
    if not get_token():
        raise SystemExit(
            "No Hugging Face token found. Run `./.venv/bin/hf auth login` "
            "or set HF_TOKEN before uploading."
        )

    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=stage_root,
        revision=revision,
        commit_message="Upload reconstructed longitudinal DCE-MRI dataset",
        ignore_patterns=[".cache/**", "**/.DS_Store"],
    )
    print(f"Upload submitted to https://huggingface.co/datasets/{repo_id}")


def main() -> None:
    args = parse_args()
    stage_root = args.stage_root.resolve()
    stage_dataset(
        stage_root=stage_root,
        link_mode=args.link_mode,
        clear=args.clear,
        dry_run=args.dry_run,
    )

    if args.upload:
        if args.dry_run:
            raise SystemExit("--dry-run and --upload cannot be used together")
        upload_dataset(stage_root=stage_root, repo_id=args.repo_id, revision=args.revision)


if __name__ == "__main__":
    main()
