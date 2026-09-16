#!/usr/bin/env python3
"""Build local release metadata for the reconstructed longitudinal dataset.

This script does not copy or modify NIfTI volumes. It scans the active
reconstructed volume folders, records quarantine exclusions, filters the
MAMA-MIA tabular data to the active reconstructed cohort, and writes a
documentation/metadata package that can later be used for GitHub or data-hosting
cleanup.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECONSTRUCTED_ROOT = PROJECT_ROOT / "Reconstructed_Datasets"
DEFAULT_MAMAMIA_CLINICAL = (
    PROJECT_ROOT / "Original_Datasets" / "MAMA-MIA" / "clinical_and_imaging_info.xlsx"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "DatasetRelease" / "reconstructed_mamamia_longitudinal"
DEFAULT_RECONSTRUCTION_AUDIT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "reconstruction_consistency_audit"
    / "reconstructed_exam_audit.csv"
)
DEFAULT_VIEW_QC = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "view_contrast_consistency"
    / "view_contrast_patient_summary.csv"
)
DEFAULT_VISUAL_QC = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "volume_visual_consistency"
    / "volume_visual_patient_summary.csv"
)

VOLUME_ROOTS = {
    "ISPY1": "ISPY1-Volumes",
    "ISPY2": "ISPY2-Volumes",
    "NACT": "NACT-Volumes",
}
SOURCE_DATASETS = set(VOLUME_ROOTS)
PHASE_RE = re.compile(r"^volume_phase(?P<phase>\d+)\.nii\.gz$")
EXAM_DATE_FORMAT = "%m-%d-%Y"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create metadata and cleaned tabular files for the reconstructed dataset."
    )
    parser.add_argument(
        "--reconstructed-root",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_ROOT,
        help="Root containing ISPY1-Volumes, ISPY2-Volumes, NACT-Volumes, and Quarantine.",
    )
    parser.add_argument(
        "--mamamia-clinical",
        type=Path,
        default=DEFAULT_MAMAMIA_CLINICAL,
        help="MAMA-MIA clinical_and_imaging_info.xlsx file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where release metadata and README are written.",
    )
    parser.add_argument(
        "--reconstruction-audit",
        type=Path,
        default=DEFAULT_RECONSTRUCTION_AUDIT,
        help="Optional reconstructed_exam_audit.csv to merge into the exam manifest.",
    )
    parser.add_argument(
        "--view-qc",
        type=Path,
        default=DEFAULT_VIEW_QC,
        help="Optional view/contrast patient QC summary to merge into the patient manifest.",
    )
    parser.add_argument(
        "--visual-qc",
        type=Path,
        default=DEFAULT_VISUAL_QC,
        help="Optional visual consistency patient QC summary to merge into the patient manifest.",
    )
    return parser.parse_args()


def canonical_patient_id(patient_id: str) -> str:
    patient_id = str(patient_id)
    if patient_id.startswith("ISPY2-"):
        return "ISPY2_" + patient_id.split("-", 1)[1]
    return patient_id


def patient_sort_key(patient_id: str) -> tuple[str, str]:
    return (canonical_patient_id(patient_id).split("_", 1)[0], canonical_patient_id(patient_id))


def parse_phase_index(path: Path) -> int | None:
    match = PHASE_RE.match(path.name)
    if not match:
        return None
    return int(match.group("phase"))


def exam_date_iso(exam_date: str) -> str:
    try:
        return datetime.strptime(exam_date, EXAM_DATE_FORMAT).date().isoformat()
    except ValueError:
        return ""


def exam_sort_key(path: Path) -> tuple[int, object]:
    try:
        return (0, datetime.strptime(path.name, EXAM_DATE_FORMAT))
    except ValueError:
        return (1, path.name)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def phase_indices_status(indices: list[int]) -> str:
    if not indices:
        return "no_phase_files"
    if indices == list(range(0, max(indices) + 1)):
        return "contiguous_from_zero"
    if indices == sorted(set(indices)):
        return "non_contiguous_or_not_zero"
    return "duplicate_or_unsorted"


def scan_active_volumes(reconstructed_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    patient_rows: list[dict[str, object]] = []
    exam_rows: list[dict[str, object]] = []
    phase_rows: list[dict[str, object]] = []

    for dataset, volume_root_name in VOLUME_ROOTS.items():
        volume_root = reconstructed_root / volume_root_name
        if not volume_root.exists():
            continue

        patient_dirs = sorted(
            [path for path in volume_root.iterdir() if path.is_dir()],
            key=lambda path: patient_sort_key(path.name),
        )
        for patient_dir in patient_dirs:
            patient_id = canonical_patient_id(patient_dir.name)
            exam_dirs = sorted(
                [path for path in patient_dir.iterdir() if path.is_dir()],
                key=exam_sort_key,
            )

            exam_dates: list[str] = []
            phase_counts: list[int] = []
            patient_phase_file_count = 0

            for timepoint_index, exam_dir in enumerate(exam_dirs):
                phase_files_with_index = [
                    (parse_phase_index(path), path)
                    for path in exam_dir.glob("volume_phase*.nii.gz")
                ]
                phase_files_with_index = [
                    (index, path)
                    for index, path in phase_files_with_index
                    if index is not None
                ]
                phase_files_with_index.sort(key=lambda item: item[0])

                indices = [int(index) for index, _ in phase_files_with_index]
                phase_files = [path for _, path in phase_files_with_index]
                exam_dates.append(exam_dir.name)
                phase_counts.append(len(phase_files))
                patient_phase_file_count += len(phase_files)

                for index, phase_path in phase_files_with_index:
                    phase_rows.append(
                        {
                            "patient_id": patient_id,
                            "reconstructed_patient_id": patient_dir.name,
                            "dataset": dataset,
                            "volume_root": volume_root_name,
                            "exam_date": exam_dir.name,
                            "exam_date_iso": exam_date_iso(exam_dir.name),
                            "timepoint_index": timepoint_index,
                            "phase_index": int(index),
                            "phase_rel_path": relative_path(phase_path),
                            "file_size_bytes": int(phase_path.stat().st_size),
                        }
                    )

                exam_rows.append(
                    {
                        "patient_id": patient_id,
                        "reconstructed_patient_id": patient_dir.name,
                        "dataset": dataset,
                        "volume_root": volume_root_name,
                        "patient_rel_path": relative_path(patient_dir),
                        "exam_date": exam_dir.name,
                        "exam_date_iso": exam_date_iso(exam_dir.name),
                        "timepoint_index": timepoint_index,
                        "exam_rel_path": relative_path(exam_dir),
                        "phase_count": len(phase_files),
                        "phase_indices": ";".join(str(index) for index in indices),
                        "phase_indices_status": phase_indices_status(indices),
                        "has_phase0": 0 in indices,
                        "first_phase_index": min(indices) if indices else pd.NA,
                        "last_phase_index": max(indices) if indices else pd.NA,
                        "phase_rel_paths": ";".join(relative_path(path) for path in phase_files),
                    }
                )

            patient_rows.append(
                {
                    "patient_id": patient_id,
                    "reconstructed_patient_id": patient_dir.name,
                    "dataset": dataset,
                    "volume_root": volume_root_name,
                    "patient_rel_path": relative_path(patient_dir),
                    "num_exams": len(exam_dirs),
                    "num_extra_exams": max(len(exam_dirs) - 1, 0),
                    "num_phase_files": patient_phase_file_count,
                    "exam_dates": ";".join(exam_dates),
                    "exam_dates_iso": ";".join(exam_date_iso(date) for date in exam_dates),
                    "phase_counts_by_exam": ";".join(str(count) for count in phase_counts),
                    "min_phase_count": min(phase_counts) if phase_counts else pd.NA,
                    "max_phase_count": max(phase_counts) if phase_counts else pd.NA,
                    "has_active_reconstructed_volumes": True,
                }
            )

    return (
        pd.DataFrame.from_records(patient_rows),
        pd.DataFrame.from_records(exam_rows),
        pd.DataFrame.from_records(phase_rows),
    )


def load_quarantine_manifest(reconstructed_root: Path) -> pd.DataFrame:
    quarantine_root = reconstructed_root / "Quarantine"
    rows: list[dict[str, object]] = []
    if not quarantine_root.exists():
        return pd.DataFrame.from_records(rows)

    for case_dir in sorted(path for path in quarantine_root.iterdir() if path.is_dir()):
        flagged_patients = case_dir / "flagged_patients.csv"
        if not flagged_patients.exists():
            continue
        df = pd.read_csv(flagged_patients)
        for row in df.to_dict(orient="records"):
            patient_id = canonical_patient_id(str(row["patient_id"]))
            rows.append(
                {
                    "patient_id": patient_id,
                    "reconstructed_patient_id": str(row["patient_id"]),
                    "quarantine_case": case_dir.name,
                    "num_flagged_exams": row.get("num_flagged_exams", ""),
                    "flagged_exam_dates": row.get("flagged_exam_dates", ""),
                    "issues": row.get("issues", ""),
                    "quarantine_rel_path": relative_path(case_dir),
                }
            )
    return pd.DataFrame.from_records(rows)


def summarize_quarantine(quarantine_manifest: pd.DataFrame) -> pd.DataFrame:
    if quarantine_manifest.empty:
        return pd.DataFrame(
            columns=[
                "patient_id",
                "quarantine_cases",
                "quarantine_flagged_exam_dates",
                "quarantine_issues",
            ]
        )

    def joined(values: Iterable[object]) -> str:
        parts = [str(value) for value in values if str(value) and str(value) != "nan"]
        return ";".join(sorted(set(parts)))

    return (
        quarantine_manifest.groupby("patient_id", dropna=False)
        .agg(
            quarantine_cases=("quarantine_case", joined),
            quarantine_flagged_exam_dates=("flagged_exam_dates", joined),
            quarantine_issues=("issues", joined),
        )
        .reset_index()
    )


def merge_reconstruction_audit(exam_manifest: pd.DataFrame, audit_path: Path) -> pd.DataFrame:
    if not audit_path.exists() or exam_manifest.empty:
        return exam_manifest

    audit = pd.read_csv(audit_path)
    audit["patient_id"] = audit["patient_id"].astype(str).map(canonical_patient_id)
    audit["exam_date"] = audit["exam_date"].astype(str)
    wanted = [
        "patient_id",
        "exam_date",
        "is_anchor_date",
        "anchor_reconstructed_date",
        "mamamia_num_phases",
        "mamamia_view",
        "metadata_exam_exists",
        "reconstructed_exam_exists",
        "reconstructed_patient_root_status",
        "quarantined_exam_exists",
        "selection_note",
        "selection_score",
        "phase_source_kind",
        "risk_status",
        "duplicate_reconstructed_phase_notes",
        "manual_phase_manifest",
        "selected_num_series",
        "selected_series_descriptions",
        "selected_num_images_metadata",
        "selected_planes",
        "expected_num_phases_from_dicom",
        "expected_phase_shapes_unique",
        "expected_phase_spacings_unique",
        "reconstructed_num_phases",
        "reconstructed_phase_indices",
        "reconstructed_phase_indices_status",
        "reconstructed_phase_shapes",
        "reconstructed_phase_spacings",
        "reconstructed_shapes_consistent",
        "reconstructed_spacings_consistent",
        "reconstructed_affines_consistent",
        "reconstructed_header_errors",
        "phase_count_matches_expected",
        "phase_header_matches_expected",
        "expected_missing_phase_indices",
        "unexpected_reconstructed_phase_indices",
        "expected_vs_reconstructed_notes",
    ]
    keep = [column for column in wanted if column in audit.columns]
    audit = audit[keep].drop_duplicates(["patient_id", "exam_date"], keep="first")
    return exam_manifest.merge(audit, on=["patient_id", "exam_date"], how="left")


def merge_patient_summary(
    patient_manifest: pd.DataFrame,
    summary_path: Path,
    prefix: str,
) -> pd.DataFrame:
    if not summary_path.exists() or patient_manifest.empty:
        return patient_manifest

    summary = pd.read_csv(summary_path)
    if "patient_id" not in summary.columns:
        return patient_manifest
    summary["patient_id"] = summary["patient_id"].astype(str).map(canonical_patient_id)

    rename = {
        column: f"{prefix}{column}"
        for column in summary.columns
        if column not in {"patient_id", "dataset"}
    }
    summary = summary.rename(columns=rename)
    keep = ["patient_id"] + list(rename.values())
    return patient_manifest.merge(summary[keep], on="patient_id", how="left")


def load_mamamia_workbook(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    clinical = pd.read_excel(path, sheet_name="dataset_info")
    legend = pd.read_excel(path, sheet_name="legend")
    clinical["patient_id"] = clinical["patient_id"].astype(str).map(canonical_patient_id)
    return clinical, legend


def add_release_status(
    clinical: pd.DataFrame,
    patient_manifest: pd.DataFrame,
    quarantine_manifest: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    active_lookup = patient_manifest.set_index("patient_id")
    active_ids = set(active_lookup.index.astype(str))
    quarantine_summary = summarize_quarantine(quarantine_manifest)
    quarantine_lookup = quarantine_summary.set_index("patient_id")
    quarantine_ids = set(quarantine_lookup.index.astype(str))

    clinical = clinical.copy()

    def status(row: pd.Series) -> str:
        patient_id = str(row["patient_id"])
        dataset = str(row["dataset"])
        if patient_id in active_ids and patient_id in quarantine_ids:
            return "active_and_quarantined"
        if patient_id in active_ids:
            return "active_reconstructed"
        if patient_id in quarantine_ids:
            return "quarantined"
        if dataset in SOURCE_DATASETS:
            return "no_active_reconstructed_volume"
        return "not_reconstructed_source_dataset"

    clinical.insert(1, "reconstructed_release_status", clinical.apply(status, axis=1))
    clinical.insert(
        2,
        "include_in_reconstructed_release",
        clinical["reconstructed_release_status"].eq("active_reconstructed"),
    )

    active_map_columns = {
        "reconstructed_patient_id": "active_reconstructed_patient_id",
        "volume_root": "active_volume_root",
        "patient_rel_path": "active_patient_rel_path",
        "num_exams": "num_reconstructed_exams",
        "num_extra_exams": "num_reconstructed_extra_exams",
        "num_phase_files": "num_reconstructed_phase_files",
        "exam_dates": "reconstructed_exam_dates",
        "exam_dates_iso": "reconstructed_exam_dates_iso",
    }
    for source_column, output_column in active_map_columns.items():
        clinical[output_column] = clinical["patient_id"].map(active_lookup[source_column])

    for column in [
        "quarantine_cases",
        "quarantine_flagged_exam_dates",
        "quarantine_issues",
    ]:
        if column in quarantine_lookup:
            clinical[column] = clinical["patient_id"].map(quarantine_lookup[column])
        else:
            clinical[column] = pd.NA

    clinical_active = clinical[clinical["include_in_reconstructed_release"]].copy()
    clinical_nonreleased = clinical[~clinical["include_in_reconstructed_release"]].copy()
    return clinical, clinical_active, clinical_nonreleased


def compact_counts(series: pd.Series) -> dict[str, int]:
    counts = series.value_counts(dropna=False).sort_index()
    return {str(key): int(value) for key, value in counts.items()}


def write_data_dictionary(path: Path) -> None:
    rows = [
        {
            "file": "patient_manifest.csv",
            "grain": "one row per active reconstructed patient",
            "description": "Active patient-level image inventory with reconstructed folder IDs, exam counts, phase counts, and patient-level QC summaries when available.",
        },
        {
            "file": "exam_manifest.csv",
            "grain": "one row per active reconstructed exam/timepoint",
            "description": "Exam-level inventory with relative paths, phase indices, and selected reconstruction audit fields.",
        },
        {
            "file": "phase_manifest.csv",
            "grain": "one row per NIfTI phase file",
            "description": "Phase-level path list for all active reconstructed volume_phase*.nii.gz files.",
        },
        {
            "file": "quarantine_manifest.csv",
            "grain": "one row per quarantined patient per quarantine case",
            "description": "Patients intentionally excluded from the active release with quarantine reason and flagged exam dates.",
        },
        {
            "file": "clinical_and_imaging_info_active_reconstructed.csv",
            "grain": "one row per active reconstructed patient",
            "description": "MAMA-MIA dataset_info rows filtered to active reconstructed patients, with release status and path summary columns added.",
        },
        {
            "file": "clinical_and_imaging_info_with_reconstruction_status.csv",
            "grain": "one row per original MAMA-MIA patient",
            "description": "Full MAMA-MIA dataset_info sheet with reconstructed_release_status added for auditability.",
        },
        {
            "file": "clinical_and_imaging_info_nonreleased.csv",
            "grain": "one row per MAMA-MIA patient not in the active reconstructed release",
            "description": "Rows excluded because they are quarantined, source patients without active reconstructed folders, or DUKE patients outside this reconstructed cohort.",
        },
    ]
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def write_readme(
    path: Path,
    summary: dict[str, object],
    output_root: Path,
    reconstructed_root: Path,
    mamamia_clinical: Path,
) -> None:
    dataset_counts = summary["active_patients_by_dataset"]
    exam_counts = summary["active_exams_by_dataset"]
    phase_counts = summary["active_phase_files_by_dataset"]
    status_counts = summary["mamamia_release_status_counts"]

    lines = [
        "# Reconstructed MAMA-MIA Longitudinal Dataset",
        "",
        "This local release package documents the reconstructed longitudinal DCE-MRI",
        "volumes derived from ISPY1, ISPY2, and Breast MRI NACT Pilot patients that",
        "also appear in MAMA-MIA. The NIfTI images are not copied into this folder;",
        "the manifests point to the active local image folders with project-relative",
        "paths.",
        "",
        "## Active Image Roots",
        "",
        f"- `Reconstructed_Datasets/ISPY1-Volumes/`: {dataset_counts.get('ISPY1', 0)} patients, {exam_counts.get('ISPY1', 0)} exams, {phase_counts.get('ISPY1', 0)} phase files",
        f"- `Reconstructed_Datasets/ISPY2-Volumes/`: {dataset_counts.get('ISPY2', 0)} patients, {exam_counts.get('ISPY2', 0)} exams, {phase_counts.get('ISPY2', 0)} phase files",
        f"- `Reconstructed_Datasets/NACT-Volumes/`: {dataset_counts.get('NACT', 0)} patients, {exam_counts.get('NACT', 0)} exams, {phase_counts.get('NACT', 0)} phase files",
        "",
        "Folder convention:",
        "",
        "```text",
        "Reconstructed_Datasets/<dataset-volume-root>/<reconstructed-patient-id>/<MM-DD-YYYY>/volume_phase<index>.nii.gz",
        "```",
        "",
        "Public-facing `patient_id` values follow MAMA-MIA naming. ISPY2 image",
        "folders use hyphens locally, for example `ISPY2-100899`, while MAMA-MIA",
        "tables use `ISPY2_100899`; both identifiers are recorded in the manifests.",
        "",
        "## Release Cohort",
        "",
        f"- Active reconstructed patients: {summary['active_patients_total']}",
        f"- Active reconstructed exams/timepoints: {summary['active_exams_total']}",
        f"- Active reconstructed NIfTI phase files: {summary['active_phase_files_total']}",
        f"- Quarantined patients excluded from active release: {summary['quarantined_patients_total']}",
        "",
        "MAMA-MIA status counts:",
        "",
        f"- `active_reconstructed`: {status_counts.get('active_reconstructed', 0)}",
        f"- `quarantined`: {status_counts.get('quarantined', 0)}",
        f"- `no_active_reconstructed_volume`: {status_counts.get('no_active_reconstructed_volume', 0)}",
        f"- `not_reconstructed_source_dataset`: {status_counts.get('not_reconstructed_source_dataset', 0)}",
        "",
        "Use `metadata/clinical_and_imaging_info_active_reconstructed.csv` as the",
        "cleaned MAMA-MIA-style tabular table for this reconstructed dataset. The",
        "full status table is kept as an audit trail.",
        "",
        "## Metadata Files",
        "",
        "- `metadata/patient_manifest.csv`: active patient-level image inventory and patient-level QC summaries.",
        "- `metadata/exam_manifest.csv`: active exam/timepoint-level inventory and reconstruction audit fields.",
        "- `metadata/phase_manifest.csv`: active phase-level NIfTI path list.",
        "- `metadata/quarantine_manifest.csv`: excluded quarantined patients and reasons.",
        "- `metadata/clinical_and_imaging_info_active_reconstructed.csv`: MAMA-MIA rows filtered to active reconstructed patients.",
        "- `metadata/clinical_and_imaging_info_with_reconstruction_status.csv`: full MAMA-MIA table with release status.",
        "- `metadata/clinical_and_imaging_info_nonreleased.csv`: MAMA-MIA rows outside the active reconstructed release.",
        "- `metadata/clinical_and_imaging_info_active_reconstructed.xlsx`: workbook copy with active, full-status, nonreleased, and legend sheets.",
        "- `metadata/release_summary.json`: machine-readable counts and provenance.",
        "- `metadata/data_dictionary.csv`: short description of each generated metadata file.",
        "",
        "## Quarantine Policy",
        "",
        "Patients in `quarantine_manifest.csv` are excluded from the active release.",
        "Current quarantine sources are geometry failures and view mismatch cases",
        "under `Reconstructed_Datasets/Quarantine/`. The original quarantined files",
        "remain on disk for audit/debugging but are not part of the active dataset.",
        "",
        "## Regeneration",
        "",
        "Run from the project root:",
        "",
        "```bash",
        "./.venv/bin/python ReconstructionScripts/build_reconstructed_dataset_release.py",
        "```",
        "",
        "Inputs used for this generated package:",
        "",
        f"- Reconstructed root: `{relative_path(reconstructed_root)}`",
        f"- MAMA-MIA clinical table: `{relative_path(mamamia_clinical)}`",
        f"- Output root: `{relative_path(output_root)}`",
        "",
        "Large NIfTI files should stay out of ordinary Git history. For a public",
        "release, keep this metadata/docs package in GitHub and place images on a",
        "data host such as Hugging Face Datasets, Git LFS, DVC, or an institutional",
        "archive.",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    reconstructed_root = args.reconstructed_root.resolve()
    output_root = args.output_root.resolve()
    metadata_root = output_root / "metadata"
    metadata_root.mkdir(parents=True, exist_ok=True)

    patient_manifest, exam_manifest, phase_manifest = scan_active_volumes(reconstructed_root)
    exam_manifest = merge_reconstruction_audit(exam_manifest, args.reconstruction_audit.resolve())
    patient_manifest = merge_patient_summary(
        patient_manifest, args.view_qc.resolve(), "view_qc_"
    )
    patient_manifest = merge_patient_summary(
        patient_manifest, args.visual_qc.resolve(), "visual_qc_"
    )

    quarantine_manifest = load_quarantine_manifest(reconstructed_root)
    clinical, legend = load_mamamia_workbook(args.mamamia_clinical.resolve())
    clinical_status, clinical_active, clinical_nonreleased = add_release_status(
        clinical=clinical,
        patient_manifest=patient_manifest,
        quarantine_manifest=quarantine_manifest,
    )

    active_ids = set(patient_manifest["patient_id"].astype(str))
    clinical_active_ids = set(clinical_active["patient_id"].astype(str))
    missing_clinical_ids = sorted(active_ids - clinical_active_ids)
    if missing_clinical_ids:
        raise RuntimeError(
            "Active reconstructed patients missing from MAMA-MIA clinical table: "
            + ", ".join(missing_clinical_ids[:20])
        )

    output_files = {
        "patient_manifest": metadata_root / "patient_manifest.csv",
        "exam_manifest": metadata_root / "exam_manifest.csv",
        "phase_manifest": metadata_root / "phase_manifest.csv",
        "quarantine_manifest": metadata_root / "quarantine_manifest.csv",
        "clinical_active": metadata_root / "clinical_and_imaging_info_active_reconstructed.csv",
        "clinical_status": metadata_root
        / "clinical_and_imaging_info_with_reconstruction_status.csv",
        "clinical_nonreleased": metadata_root / "clinical_and_imaging_info_nonreleased.csv",
        "clinical_workbook": metadata_root
        / "clinical_and_imaging_info_active_reconstructed.xlsx",
        "data_dictionary": metadata_root / "data_dictionary.csv",
        "summary": metadata_root / "release_summary.json",
        "readme": output_root / "README.md",
    }

    patient_manifest.to_csv(output_files["patient_manifest"], index=False)
    exam_manifest.to_csv(output_files["exam_manifest"], index=False)
    phase_manifest.to_csv(output_files["phase_manifest"], index=False)
    quarantine_manifest.to_csv(output_files["quarantine_manifest"], index=False)
    clinical_active.to_csv(output_files["clinical_active"], index=False)
    clinical_status.to_csv(output_files["clinical_status"], index=False)
    clinical_nonreleased.to_csv(output_files["clinical_nonreleased"], index=False)
    write_data_dictionary(output_files["data_dictionary"])

    with pd.ExcelWriter(output_files["clinical_workbook"], engine="openpyxl") as writer:
        clinical_active.to_excel(writer, sheet_name="active_reconstructed", index=False)
        clinical_status.to_excel(writer, sheet_name="all_with_status", index=False)
        clinical_nonreleased.to_excel(writer, sheet_name="nonreleased", index=False)
        legend.to_excel(writer, sheet_name="legend", index=False)

    summary: dict[str, object] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_root": relative_path(PROJECT_ROOT),
        "reconstructed_root": relative_path(reconstructed_root),
        "mamamia_clinical": relative_path(args.mamamia_clinical.resolve()),
        "output_root": relative_path(output_root),
        "active_patients_total": int(len(patient_manifest)),
        "active_exams_total": int(len(exam_manifest)),
        "active_phase_files_total": int(len(phase_manifest)),
        "quarantined_patients_total": int(quarantine_manifest["patient_id"].nunique())
        if not quarantine_manifest.empty
        else 0,
        "active_patients_by_dataset": compact_counts(patient_manifest["dataset"]),
        "active_exams_by_dataset": compact_counts(exam_manifest["dataset"]),
        "active_phase_files_by_dataset": compact_counts(phase_manifest["dataset"]),
        "mamamia_release_status_counts": compact_counts(
            clinical_status["reconstructed_release_status"]
        ),
        "exam_phase_count_distribution": compact_counts(exam_manifest["phase_count"]),
        "files_written": {
            key: relative_path(value)
            for key, value in output_files.items()
            if key != "readme"
        },
    }
    output_files["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_readme(
        path=output_files["readme"],
        summary=summary,
        output_root=output_root,
        reconstructed_root=reconstructed_root,
        mamamia_clinical=args.mamamia_clinical.resolve(),
    )

    print(f"Wrote release package: {relative_path(output_root)}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
