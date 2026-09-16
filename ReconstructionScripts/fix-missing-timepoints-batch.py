import argparse
import csv
import importlib.util
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_PATH = SCRIPT_DIR / "make-volumes-ispy2.py"

PATTERN_MATCH_RULES = {
    "later_original_dce_family": ["original dce"],
    "later_dyn_mp_family": ["dyn mp", "bw83 dyn"],
}


def load_main_module():
    spec = importlib.util.spec_from_file_location("make_volumes_ispy2", MAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_report_df(report_csv: Path) -> pd.DataFrame:
    expected_columns = [
        "workbook_patient_id",
        "patient_id",
        "anchor_uid",
        "anchor_series_description",
        "exam_date",
        "study_description",
        "exam_dir",
        "selected_series_uids",
        "selected_series_descriptions",
        "selected_series_dirs",
        "selection_note",
        "selection_score",
        "phase_source_kind",
        "phase_index",
        "num_selected_series",
        "num_dicom_files",
        "num_phases",
        "num_slices_in_phase",
        "output_path",
        "status",
        "message",
    ]
    if report_csv.exists():
        report_df = pd.read_csv(report_csv)
    else:
        report_df = pd.DataFrame(columns=expected_columns)

    for column in expected_columns:
        if column not in report_df.columns:
            report_df[column] = ""

    return report_df[expected_columns].copy()


def replace_report_rows(
    report_df: pd.DataFrame,
    *,
    patient_id: str,
    exam_date: str,
    new_rows: Sequence[Dict[str, object]],
) -> pd.DataFrame:
    keep_mask = ~(
        (report_df["patient_id"].astype(str) == patient_id)
        & (report_df["exam_date"].astype(str) == exam_date)
    )
    report_df = report_df.loc[keep_mask].copy()
    replacement_df = pd.DataFrame(list(new_rows), columns=report_df.columns)
    report_df = pd.concat([report_df, replacement_df], ignore_index=True)
    return report_df


def build_report_rows(
    *,
    workbook_patient_id: str,
    patient_id: str,
    anchor_uid: str,
    anchor_series_description: str,
    exam_date: str,
    study_description: str,
    exam_dir: Path,
    selected_candidates: Sequence,
    phases: Sequence[Sequence[Path]],
    output_root: Path,
    dry_run: bool,
    phase_source_kind: str,
    selection_note: str,
) -> List[Dict[str, object]]:
    rows = []
    selected_series_uids = " | ".join(candidate.series_uid for candidate in selected_candidates)
    selected_series_descriptions = " | ".join(candidate.series_description for candidate in selected_candidates)
    selected_series_dirs = " | ".join(str(candidate.series_dir) for candidate in selected_candidates)
    total_dicom_files = sum(len(phase_files) for phase_files in phases)

    for phase_index, phase_files in enumerate(phases):
        output_path = output_root / patient_id / exam_date / f"volume_phase{phase_index}.nii.gz"
        rows.append(
            {
                "workbook_patient_id": workbook_patient_id,
                "patient_id": patient_id,
                "anchor_uid": anchor_uid,
                "anchor_series_description": anchor_series_description,
                "exam_date": exam_date,
                "study_description": study_description,
                "exam_dir": str(exam_dir),
                "selected_series_uids": selected_series_uids,
                "selected_series_descriptions": selected_series_descriptions,
                "selected_series_dirs": selected_series_dirs,
                "selection_note": selection_note,
                "selection_score": "batch_manual_override",
                "phase_source_kind": phase_source_kind,
                "phase_index": phase_index,
                "num_selected_series": len(selected_candidates),
                "num_dicom_files": total_dicom_files,
                "num_phases": len(phases),
                "num_slices_in_phase": len(phase_files),
                "output_path": str(output_path),
                "status": "dry_run" if dry_run else "written",
                "message": (
                    "Dry run enabled; batch override report updated without writing NIfTI files"
                    if dry_run
                    else "Batch override applied and NIfTI volume written successfully"
                ),
            }
        )

    return rows


def find_matching_candidates(exam_candidates, match_terms: Sequence[str]):
    for term in match_terms:
        lowered = term.lower()
        matches = [
            candidate
            for candidate in exam_candidates
            if lowered in candidate.series_description.lower() or lowered in (candidate.family_key or "").lower()
        ]
        if matches:
            return matches, term
    return [], ""


def derive_phases(main_module, selected_candidates, phase_gap: int):
    ordered = main_module.sort_candidates_by_phase(selected_candidates)
    if len(ordered) == 1:
        dicom_files = main_module.list_dicom_files_sorted(ordered[0].series_dir)
        phases, auto_source_kind = main_module.split_single_series_into_phases(dicom_files, phase_gap)
        return ordered, phases, auto_source_kind

    phases = []
    for candidate in ordered:
        dicom_files = main_module.list_dicom_files_sorted(candidate.series_dir)
        if dicom_files:
            phases.append(dicom_files)
    return ordered, phases, "multi_series_batch_family"


def ensure_report_backup(report_csv: Path) -> Path:
    backup_path = report_csv.with_name(report_csv.stem + ".before_missing_timepoint_batch_fix.csv")
    if report_csv.exists() and not backup_path.exists():
        shutil.copy2(report_csv, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fix missing reconstructed ISPY-2 timepoints by reading the missing-timepoint analysis, "
            "selecting the appropriate later dynamic family, reconstructing the missing exams, and "
            "updating the report."
        )
    )
    parser.add_argument(
        "--analysis_csv",
        type=Path,
        default=SCRIPT_DIR / "ispy2_missing_timepoints_analysis.csv",
    )
    parser.add_argument(
        "--summary_csv",
        type=Path,
        default=SCRIPT_DIR / "ispy2_missing_timepoints_fixed_summary.csv",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=SCRIPT_DIR / "ispy2_volume_construction_report.csv",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("Reconstructed_Datasets/ISPY2-Volumes"),
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["later_original_dce_family", "later_dyn_mp_family"],
    )
    parser.add_argument(
        "--phase_gap",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
    )
    args = parser.parse_args()

    main_module = load_main_module()

    analysis_rows = list(csv.DictReader(args.analysis_csv.open(newline="")))
    requested_patterns = set(args.patterns)
    target_rows = [row for row in analysis_rows if row["likely_pattern"] in requested_patterns]

    excel_df = pd.read_excel(main_module.DEFAULT_EXCEL_PATH, sheet_name="dataset_info")
    excel_df["dataset"] = excel_df["dataset"].astype(str).str.strip()
    excel_df["patient_id"] = excel_df["patient_id"].astype(str).str.strip()
    excel_df["tcia_series_uid"] = excel_df["tcia_series_uid"].astype(str).str.strip()
    excel_df["normalized_patient_id"] = excel_df["patient_id"].map(main_module.normalize_ispy2_patient_id)
    excel_df = excel_df[excel_df["dataset"] == "ISPY2"].copy()
    workbook_by_patient = {row["normalized_patient_id"]: row for _, row in excel_df.iterrows()}

    metadata_by_patient, metadata_by_uid = main_module.build_metadata_candidates(main_module.DEFAULT_CSV_PATH)
    exam_groups_cache = {patient_id: main_module.group_patient_exams(cands) for patient_id, cands in metadata_by_patient.items()}

    report_df = load_report_df(args.report_csv)
    backup_path = None
    if not args.dry_run:
        backup_path = ensure_report_backup(args.report_csv)

    summary_rows = []
    fixed_count = 0
    skipped = []

    for index, row in enumerate(target_rows, start=1):
        patient_id = row["patient_id"]
        exam_date = row["missing_date"]
        pattern = row["likely_pattern"]
        print(f"[{index}/{len(target_rows)}] {patient_id} {exam_date} {pattern}", flush=True)

        workbook_row = workbook_by_patient.get(patient_id)
        if workbook_row is None:
            skipped.append((patient_id, exam_date, "workbook_row_missing"))
            continue

        anchor_uid = main_module.clean_uid(workbook_row["tcia_series_uid"])
        anchor_candidate = metadata_by_uid.get(anchor_uid)
        if anchor_candidate is None:
            skipped.append((patient_id, exam_date, "anchor_uid_missing_in_metadata"))
            continue

        exam_groups = exam_groups_cache.get(patient_id, {})
        exam_candidates = None
        for candidates in exam_groups.values():
            candidate_date = main_module.extract_exam_date_from_exam_folder(candidates[0].exam_dir.name)
            if candidate_date == exam_date:
                exam_candidates = candidates
                break

        if exam_candidates is None:
            skipped.append((patient_id, exam_date, "exam_group_missing"))
            continue

        match_terms = PATTERN_MATCH_RULES.get(pattern, [])
        matched_candidates, matched_term = find_matching_candidates(exam_candidates, match_terms)
        if not matched_candidates:
            skipped.append((patient_id, exam_date, f"no_series_match_for_pattern:{pattern}"))
            continue

        selected_candidates, phases, auto_source_kind = derive_phases(main_module, matched_candidates, args.phase_gap)
        if not phases:
            skipped.append((patient_id, exam_date, f"no_phases_from_selected_family:{pattern}"))
            continue

        patient_out_dir = args.output_root / patient_id / exam_date
        if not args.dry_run:
            patient_out_dir.mkdir(parents=True, exist_ok=True)
            for old_volume in patient_out_dir.glob("volume_phase*.nii.gz"):
                old_volume.unlink()
            for phase_index, phase_files in enumerate(phases):
                image = main_module.build_sitk_image_from_phase_files(phase_files)
                output_path = patient_out_dir / f"volume_phase{phase_index}.nii.gz"
                main_module.sitk.WriteImage(image, str(output_path))

        phase_source_kind = f"batch_override_{auto_source_kind}"
        selection_note = (
            f"batch_override_pattern={pattern}; "
            f"series_match_term={matched_term}; "
            f"selected_family={selected_candidates[0].family_key or selected_candidates[0].series_description.lower()}; "
            f"natural_phase_split={auto_source_kind}"
        )
        report_rows = build_report_rows(
            workbook_patient_id=str(workbook_row["patient_id"]).strip(),
            patient_id=patient_id,
            anchor_uid=anchor_uid,
            anchor_series_description=anchor_candidate.series_description,
            exam_date=exam_date,
            study_description=selected_candidates[0].study_description,
            exam_dir=selected_candidates[0].exam_dir,
            selected_candidates=selected_candidates,
            phases=phases,
            output_root=args.output_root,
            dry_run=args.dry_run,
            phase_source_kind=phase_source_kind,
            selection_note=selection_note,
        )
        report_df = replace_report_rows(
            report_df,
            patient_id=patient_id,
            exam_date=exam_date,
            new_rows=report_rows,
        )

        fixed_count += 1
        summary_rows.append(
            {
                "patient_id": patient_id,
                "exam_date": exam_date,
                "pattern": pattern,
                "series_match_term": matched_term,
                "selected_series_uids": " | ".join(candidate.series_uid for candidate in selected_candidates),
                "selected_series_descriptions": " | ".join(candidate.series_description for candidate in selected_candidates),
                "selected_series_dirs": " | ".join(str(candidate.series_dir) for candidate in selected_candidates),
                "num_phases": len(phases),
                "phase_sizes": " | ".join(str(len(phase_files)) for phase_files in phases),
                "phase_source_kind": phase_source_kind,
                "status": "dry_run" if args.dry_run else "written",
            }
        )

    if not args.dry_run:
        args.report_csv.parent.mkdir(parents=True, exist_ok=True)
        report_df = report_df.sort_values(["patient_id", "exam_date", "phase_index"], kind="stable").reset_index(drop=True)
        report_df.to_csv(args.report_csv, index=False)

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["patient_id", "exam_date"], kind="stable").reset_index(drop=True)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"Target missing dates: {len(target_rows)}")
    print(f"Fixed dates: {fixed_count}")
    print(f"Skipped dates: {len(skipped)}")
    if backup_path is not None:
        print(f"Report backup: {backup_path}")
    print(f"Summary CSV: {args.summary_csv}")
    if skipped:
        for patient_id, exam_date, reason in skipped[:20]:
            print(f"SKIP {patient_id} {exam_date} {reason}")
        if len(skipped) > 20:
            print(f"... {len(skipped) - 20} more skipped dates")


if __name__ == "__main__":
    main()
