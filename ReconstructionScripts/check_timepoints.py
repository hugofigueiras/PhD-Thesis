import argparse
from pathlib import Path
import re
from typing import Dict, List, Set

import pandas as pd


def extract_patient_id_from_target(patient_dir_name: str) -> str:
    """
    Example:
      UCSF-BR-20 -> NACT_20
    """
    m = re.match(r"UCSF-BR-(\d+)", patient_dir_name, flags=re.IGNORECASE)
    if m:
        return f"NACT_{int(m.group(1)):02d}"
    return patient_dir_name


def extract_date_from_target_exam_dir(exam_dir_name: str) -> str:
    """
    Examples:
      01-09-1992-205329-MR BREAS UNIT-12345 -> 01-09-1992
      03-16-1992-xxxx -> 03-16-1992
    """
    m = re.match(r"^(\d{2}-\d{2}-\d{4})", exam_dir_name)
    if m:
        return m.group(1)
    return ""


def collect_target_timepoints(target_root: Path) -> Dict[str, Set[str]]:
    """
    Expected target structure:
      target_root/
        UCSF-BR-01/
          03-29-1990-.../
          04-17-1990-.../
        UCSF-BR-20/
          12-17-1991-.../
          01-09-1992-.../
    """
    result: Dict[str, Set[str]] = {}

    for patient_dir in sorted(target_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = extract_patient_id_from_target(patient_dir.name)
        dates = set()

        for exam_dir in sorted(patient_dir.iterdir()):
            if not exam_dir.is_dir():
                continue

            date_str = extract_date_from_target_exam_dir(exam_dir.name)
            if date_str:
                dates.add(date_str)

        if dates:
            result[patient_id] = dates

    return result


def collect_generated_timepoints(generated_root: Path) -> Dict[str, Set[str]]:
    """
    Expected generated structure:
      generated_root/
        NACT_01/
          03-29-1990/
          04-17-1990/
        NACT_20/
          12-17-1991/
          01-09-1992/
    """
    result: Dict[str, Set[str]] = {}

    for patient_dir in sorted(generated_root.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name.strip()
        dates = set()

        for date_dir in sorted(patient_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            if re.match(r"^\d{2}-\d{2}-\d{4}$", date_dir.name):
                dates.add(date_dir.name)

        if dates:
            result[patient_id] = dates
        else:
            result.setdefault(patient_id, set())

    return result


def main():
    ap = argparse.ArgumentParser(
        description="Check whether the generated dataset has the same time points as the target dataset."
    )
    ap.add_argument(
        "--target_root",
        type=Path,
        required=True,
        help="Root of the original target dataset (contains UCSF-BR-XX folders)."
    )
    ap.add_argument(
        "--generated_root",
        type=Path,
        required=True,
        help="Root of the generated dataset (contains NACT_XX folders)."
    )
    ap.add_argument(
        "--report_csv",
        type=Path,
        default=Path("timepoint_comparison_report.csv"),
        help="Output CSV report path."
    )

    args = ap.parse_args()

    target_map = collect_target_timepoints(args.target_root)
    generated_map = collect_generated_timepoints(args.generated_root)

    all_patients = sorted(set(target_map.keys()) | set(generated_map.keys()))
    rows: List[dict] = []

    for patient_id in all_patients:
        target_dates = target_map.get(patient_id, set())
        generated_dates = generated_map.get(patient_id, set())

        missing_dates = sorted(target_dates - generated_dates)
        extra_dates = sorted(generated_dates - target_dates)

        rows.append({
            "patient_id": patient_id,
            "num_target_timepoints": len(target_dates),
            "num_generated_timepoints": len(generated_dates),
            "target_timepoints": ";".join(sorted(target_dates)),
            "generated_timepoints": ";".join(sorted(generated_dates)),
            "missing_in_generated": ";".join(missing_dates),
            "extra_in_generated": ";".join(extra_dates),
            "same_timepoints": target_dates == generated_dates,
        })

    df = pd.DataFrame(rows)
    args.report_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.report_csv, index=False)

    total = len(df)
    exact = int(df["same_timepoints"].sum()) if total > 0 else 0

    print(f"Patients checked: {total}")
    print(f"Patients with exact matching time points: {exact}")
    print(f"CSV report written to: {args.report_csv}")


if __name__ == "__main__":
    main()