from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXAM_REPORT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "reconstructed_volumes_header"
    / "exam_report.csv"
)
DEFAULT_RECONSTRUCTED_ROOT = PROJECT_ROOT / "Reconstructed_Datasets"
DEFAULT_QUARANTINE_ROOT = (
    DEFAULT_RECONSTRUCTED_ROOT / "Quarantine" / "ISPY1-Geometry-Flagged-Patients"
)

SOURCE_DIRS = (
    "ISPY1-Volumes",
    "MAMA-MIA-Masks",
    "Original-VOIs",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move flagged ISPY1 patients out of active reconstructed datasets into "
            "a dedicated quarantine area."
        )
    )
    parser.add_argument(
        "--exam-report",
        type=Path,
        default=DEFAULT_EXAM_REPORT,
        help="QC exam report used to identify flagged ISPY1 patients.",
    )
    parser.add_argument(
        "--reconstructed-root",
        type=Path,
        default=DEFAULT_RECONSTRUCTED_ROOT,
        help="Root of Reconstructed_Datasets.",
    )
    parser.add_argument(
        "--quarantine-root",
        type=Path,
        default=DEFAULT_QUARANTINE_ROOT,
        help="Destination root for quarantined patient folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be moved.",
    )
    return parser.parse_args()


def load_flagged_patients(exam_report_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    exam_df = pd.read_csv(exam_report_path)
    flagged_exam_df = exam_df[
        (exam_df["dataset_dir"] == "ISPY1-Volumes") & (exam_df["issue_count"] > 0)
    ].copy()
    flagged_exam_df = flagged_exam_df.sort_values(
        ["patient_id", "acquisition_date"], kind="stable"
    )

    patient_df = (
        flagged_exam_df.groupby("patient_id", dropna=False)
        .agg(
            num_flagged_exams=("acquisition_date", "nunique"),
            flagged_exam_dates=("acquisition_date", lambda s: ";".join(sorted(set(s.astype(str))))),
            issues=("issues", lambda s: "|".join(sorted(set(s.astype(str))))),
        )
        .reset_index()
        .sort_values("patient_id", kind="stable")
    )
    return flagged_exam_df, patient_df


def move_one(source: Path, destination: Path, dry_run: bool) -> str:
    if not source.exists():
        return "missing_source"
    if destination.exists():
        return "destination_exists"
    if dry_run:
        return "dry_run"

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return "moved"


def main() -> None:
    args = parse_args()
    exam_report_path = args.exam_report.resolve()
    reconstructed_root = args.reconstructed_root.resolve()
    quarantine_root = args.quarantine_root.resolve()

    flagged_exam_df, patient_df = load_flagged_patients(exam_report_path)

    manifest_rows: list[dict[str, str]] = []
    for patient in patient_df["patient_id"]:
        for source_dir_name in SOURCE_DIRS:
            source = reconstructed_root / source_dir_name / patient
            destination = quarantine_root / source_dir_name / patient
            status = move_one(source=source, destination=destination, dry_run=args.dry_run)
            manifest_rows.append(
                {
                    "patient_id": str(patient),
                    "source_group": source_dir_name,
                    "source_path": str(source),
                    "destination_path": str(destination),
                    "status": status,
                }
            )

    quarantine_root.mkdir(parents=True, exist_ok=True)
    flagged_exam_df.to_csv(quarantine_root / "flagged_exams.csv", index=False)
    patient_df.to_csv(quarantine_root / "flagged_patients.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(quarantine_root / "move_manifest.csv", index=False)

    summary_lines = [
        "ISPY1 geometry quarantine",
        f"Exam report: {exam_report_path}",
        f"Patients flagged: {len(patient_df)}",
        f"Flagged exams: {len(flagged_exam_df)}",
        f"Dry run: {args.dry_run}",
        "",
        "Patients:",
    ]
    for row in patient_df.itertuples(index=False):
        summary_lines.append(
            f"- {row.patient_id}: {row.flagged_exam_dates} [{row.issues}]"
        )
    (quarantine_root / "README.txt").write_text("\n".join(summary_lines) + "\n")

    manifest_df = pd.DataFrame(manifest_rows)
    print(f"Flagged patients: {len(patient_df)}")
    print(f"Flagged exams: {len(flagged_exam_df)}")
    print(manifest_df["status"].value_counts().to_string())
    print(f"Quarantine root: {quarantine_root}")


if __name__ == "__main__":
    main()
