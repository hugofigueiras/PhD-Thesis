import argparse
import importlib.util
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT_PATH = SCRIPT_DIR / "make-volumes-ispy2.py"


def load_main_module():
    spec = importlib.util.spec_from_file_location("make_volumes_ispy2", MAIN_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def find_matching_candidates(exam_candidates, series_matches: Sequence[str]):
    normalized_matches = [match.lower() for match in series_matches if match]
    matching_candidates = [
        candidate
        for candidate in exam_candidates
        if any(match in candidate.series_description.lower() for match in normalized_matches)
    ]
    if not matching_candidates:
        return []
    return matching_candidates


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
                "selection_score": "manual_override",
                "phase_source_kind": phase_source_kind,
                "phase_index": phase_index,
                "num_selected_series": len(selected_candidates),
                "num_dicom_files": total_dicom_files,
                "num_phases": len(phases),
                "num_slices_in_phase": len(phase_files),
                "output_path": str(output_path),
                "status": "dry_run" if dry_run else "written",
                "message": (
                    "Dry run enabled; override report updated without writing NIfTI files"
                    if dry_run
                    else "Manual override applied and NIfTI volume written successfully"
                ),
            }
        )

    return rows


def update_report_csv(report_csv: Path, patient_id: str, exam_date: str, new_rows: Sequence[Dict[str, object]]) -> None:
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

    keep_mask = ~(
        (report_df["patient_id"].astype(str) == patient_id)
        & (report_df["exam_date"].astype(str) == exam_date)
    )
    report_df = report_df.loc[keep_mask, expected_columns].copy()

    override_df = pd.DataFrame(list(new_rows), columns=expected_columns)
    report_df = pd.concat([report_df, override_df], ignore_index=True)
    report_df = report_df.sort_values(["patient_id", "exam_date", "phase_index"], kind="stable").reset_index(drop=True)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Manually override ISPY-2 exams by selecting a specific series family and reconstructing "
            "its natural phases directly from that series or series family. Defaults to applying "
            "the original DCE override across all dates for ISPY2-778020."
        )
    )
    parser.add_argument("--patient_id", default="ISPY2-778020")
    parser.add_argument("--exam_date", default=None)
    parser.add_argument("--all_dates", action="store_true", default=True)
    parser.add_argument(
        "--series_match",
        nargs="+",
        default=["original dce"],
    )
    parser.add_argument("--phase_gap", type=int, default=80)
    parser.add_argument("--excel_path", type=Path, default=None)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--report_csv", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    main_module = load_main_module()
    excel_path = args.excel_path or main_module.DEFAULT_EXCEL_PATH
    csv_path = args.csv_path or main_module.DEFAULT_CSV_PATH
    output_root = args.output_root or main_module.DEFAULT_OUTPUT_ROOT
    report_csv = args.report_csv or main_module.DEFAULT_REPORT_CSV
    series_matches = [str(match).strip() for match in args.series_match if str(match).strip()]

    patient_id = main_module.normalize_ispy2_patient_id(args.patient_id)
    excel_df = pd.read_excel(excel_path, sheet_name="dataset_info")
    excel_df["dataset"] = excel_df["dataset"].astype(str).str.strip()
    excel_df["patient_id"] = excel_df["patient_id"].astype(str).str.strip()
    excel_df["tcia_series_uid"] = excel_df["tcia_series_uid"].astype(str).str.strip()
    excel_df["normalized_patient_id"] = excel_df["patient_id"].map(main_module.normalize_ispy2_patient_id)

    workbook_rows = excel_df[
        (excel_df["dataset"] == "ISPY2")
        & (excel_df["normalized_patient_id"] == patient_id)
    ]
    if workbook_rows.empty:
        raise ValueError(f"No ISPY2 workbook row found for patient {patient_id}")

    workbook_row = workbook_rows.iloc[0]
    workbook_patient_id = str(workbook_row["patient_id"]).strip()
    anchor_uid = main_module.clean_uid(workbook_row["tcia_series_uid"])

    metadata_by_patient, metadata_by_uid = main_module.build_metadata_candidates(csv_path)
    anchor_candidate = metadata_by_uid.get(anchor_uid)
    if anchor_candidate is None:
        raise ValueError(f"Anchor UID {anchor_uid} was not found in metadata.csv")

    exam_groups = main_module.group_patient_exams(metadata_by_patient.get(patient_id, []))
    ordered_exam_groups = sorted(
        exam_groups.values(),
        key=lambda candidates: (
            main_module.extract_exam_date_from_exam_folder(candidates[0].exam_dir.name) == "unknown_date",
            main_module.extract_exam_date_from_exam_folder(candidates[0].exam_dir.name),
        ),
    )

    exams_to_process = []
    for candidates in ordered_exam_groups:
        exam_date = main_module.extract_exam_date_from_exam_folder(candidates[0].exam_dir.name)
        if args.exam_date and exam_date != args.exam_date:
            continue
        exams_to_process.append(candidates)

    if not exams_to_process:
        raise ValueError(f"No matching exams found for patient {patient_id}")

    processed_dates = []
    skipped_dates = []
    for exam_candidates in exams_to_process:
        exam_date = main_module.extract_exam_date_from_exam_folder(exam_candidates[0].exam_dir.name)
        matched_candidates = find_matching_candidates(exam_candidates, series_matches)
        if not matched_candidates:
            skipped_dates.append(exam_date)
            continue
        selected_candidates = main_module.sort_candidates_by_phase(matched_candidates)
        if len(selected_candidates) == 1:
            dicom_files = main_module.list_dicom_files_sorted(selected_candidates[0].series_dir)
            phases, auto_source_kind = main_module.split_single_series_into_phases(dicom_files, args.phase_gap)
            if not phases:
                raise ValueError(
                    f"Could not derive phases from {selected_candidates[0].series_dir} for {patient_id} on {exam_date}"
                )
        else:
            phases = []
            for candidate in selected_candidates:
                dicom_files = main_module.list_dicom_files_sorted(candidate.series_dir)
                if not dicom_files:
                    continue
                phases.append(dicom_files)
            if not phases:
                raise ValueError(
                    f"No DICOM files found for matched family {series_matches!r} on {patient_id} {exam_date}"
                )
            auto_source_kind = "multi_series_manual_family"

        patient_out_dir = output_root / patient_id / exam_date
        if not args.dry_run:
            patient_out_dir.mkdir(parents=True, exist_ok=True)
            for old_volume in patient_out_dir.glob("volume_phase*.nii.gz"):
                old_volume.unlink()

            for phase_index, phase_files in enumerate(phases):
                image = main_module.build_sitk_image_from_phase_files(phase_files)
                output_path = patient_out_dir / f"volume_phase{phase_index}.nii.gz"
                main_module.sitk.WriteImage(image, str(output_path))

        phase_source_kind = f"manual_override_{auto_source_kind}"
        selection_note = (
            f"manual_override_series_match={' | '.join(series_matches)}; "
            f"selected_family={selected_candidates[0].family_key or selected_candidates[0].series_description.lower()}; "
            f"natural_phase_split={auto_source_kind}"
        )
        report_rows = build_report_rows(
            workbook_patient_id=workbook_patient_id,
            patient_id=patient_id,
            anchor_uid=anchor_uid,
            anchor_series_description=anchor_candidate.series_description,
            exam_date=exam_date,
            study_description=selected_candidates[0].study_description,
            exam_dir=selected_candidates[0].exam_dir,
            selected_candidates=selected_candidates,
            phases=phases,
            output_root=output_root,
            dry_run=args.dry_run,
            phase_source_kind=phase_source_kind,
            selection_note=selection_note,
        )
        update_report_csv(report_csv, patient_id, exam_date, report_rows)

        processed_dates.append(
            {
                "exam_date": exam_date,
                "series_uid": " | ".join(candidate.series_uid for candidate in selected_candidates),
                "series_description": " | ".join(candidate.series_description for candidate in selected_candidates),
                "series_dir": " | ".join(str(candidate.series_dir) for candidate in selected_candidates),
                "num_phases": len(phases),
                "phase_sizes": [len(phase_files) for phase_files in phases],
                "phase_source_kind": phase_source_kind,
            }
        )

    print(f"Patient: {patient_id}")
    for item in processed_dates:
        print(f"Exam date: {item['exam_date']}")
        print(f"  Selected series UID: {item['series_uid']}")
        print(f"  Selected series description: {item['series_description']}")
        print(f"  Selected series dir: {item['series_dir']}")
        print(f"  Phases written: {item['num_phases']}")
        print(f"  Slices per phase: {item['phase_sizes']}")
        print(f"  Phase source kind: {item['phase_source_kind']}")
    if skipped_dates:
        print(f"Skipped dates with no matching family: {skipped_dates}")
    if args.dry_run:
        print("Dry run enabled: NIfTI files were not written.")
    print(f"Report updated: {report_csv}")


if __name__ == "__main__":
    main()
