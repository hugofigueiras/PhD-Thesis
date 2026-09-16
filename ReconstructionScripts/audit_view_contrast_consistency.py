#!/usr/bin/env python3
"""Audit view and T1/T2-like contrast consistency across reconstructed exams."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECON_AUDIT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "reconstruction_consistency_audit"
    / "reconstructed_exam_audit.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "view_contrast_consistency"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether selected source MRI series keep the same view/plane and "
            "T1/T2-like contrast class as the MAMA-MIA anchor time point."
        )
    )
    parser.add_argument("--reconstruction-audit-csv", type=Path, default=DEFAULT_RECON_AUDIT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--include-inactive",
        action="store_true",
        help="Also include rows without an active reconstructed exam.",
    )
    return parser.parse_args()


def split_pipe(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def parse_float(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        parsed = float(value)
    except Exception:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def list_dicom_files(series_dir: Path) -> list[Path]:
    try:
        files = [
            path
            for path in series_dir.iterdir()
            if path.is_file()
            and not path.name.lower().endswith(
                (".nii", ".nii.gz", ".csv", ".json", ".txt", ".tsv")
            )
        ]
    except FileNotFoundError:
        return []
    return sorted(files, key=lambda path: path.name)


def first_reconstructed_phase(exam_dir_text: object) -> Path | None:
    if exam_dir_text is None or pd.isna(exam_dir_text):
        return None
    exam_dir = Path(str(exam_dir_text))
    if not exam_dir.is_dir():
        return None
    phase_paths = sorted(exam_dir.glob("volume_phase*.nii.gz"))
    return phase_paths[0] if phase_paths else None


def infer_view_from_nifti(path: Path | None) -> dict[str, object]:
    if path is None:
        return {
            "nifti_header_path": "",
            "nifti_header_view": "",
            "nifti_header_axcodes": "",
            "nifti_header_load_error": "",
        }
    try:
        image = nib.load(str(path))
        affine = np.asarray(image.affine, dtype=float)
        axcodes = "".join(nib.orientations.aff2axcodes(affine))
        slice_vector = affine[:3, 2]
        axis = int(np.argmax(np.abs(slice_vector)))
        view = ("sagittal", "coronal", "axial")[axis]
        return {
            "nifti_header_path": str(path),
            "nifti_header_view": view,
            "nifti_header_axcodes": axcodes,
            "nifti_header_load_error": "",
        }
    except Exception as exc:
        return {
            "nifti_header_path": str(path),
            "nifti_header_view": "",
            "nifti_header_axcodes": "",
            "nifti_header_load_error": str(exc),
        }


@lru_cache(maxsize=30000)
def read_first_header(series_dir_text: str) -> pydicom.Dataset | None:
    files = list_dicom_files(Path(series_dir_text))
    if not files:
        return None
    try:
        return pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return None


def dicom_text(ds: pydicom.Dataset | None) -> str:
    if ds is None:
        return ""
    fields = []
    for attr in (
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "MRAcquisitionType",
    ):
        value = getattr(ds, attr, None)
        if value is not None:
            fields.append(str(value))
    return " | ".join(fields)


def plane_from_iop(value: object) -> str:
    try:
        if value is None or len(value) != 6:
            return ""
        row_cos = np.array([float(item) for item in value[:3]], dtype=float)
        col_cos = np.array([float(item) for item in value[3:]], dtype=float)
        normal = np.cross(row_cos, col_cos)
        axis = int(np.argmax(np.abs(normal)))
    except Exception:
        return ""
    if axis == 0:
        return "sagittal"
    if axis == 1:
        return "coronal"
    return "axial"


def infer_plane(series_description: str, ds: pydicom.Dataset | None) -> str:
    text = f"{series_description} | {dicom_text(ds)}".lower()
    if re.search(r"\bsag(it+al)?\b", text):
        return "sagittal"
    if re.search(r"\b(ax|axial|tra|transverse)\b", text):
        return "axial"
    if re.search(r"\bcor(on?al)?\b", text):
        return "coronal"
    if ds is not None:
        return plane_from_iop(getattr(ds, "ImageOrientationPatient", None))
    return ""


def numeric_header_values(ds: pydicom.Dataset | None) -> dict[str, float | None]:
    values: dict[str, float | None] = {}
    if ds is None:
        return {"tr": None, "te": None, "ti": None, "flip_angle": None}
    mapping = {
        "tr": "RepetitionTime",
        "te": "EchoTime",
        "ti": "InversionTime",
        "flip_angle": "FlipAngle",
    }
    for key, attr in mapping.items():
        values[key] = parse_float(getattr(ds, attr, None))
    return values


def infer_contrast_class(series_description: str, ds: pydicom.Dataset | None) -> tuple[str, str]:
    text = f"{series_description} | {dicom_text(ds)}".lower()
    values = numeric_header_values(ds)
    tr = values["tr"]
    te = values["te"]
    reasons = []

    if re.search(r"\b(sub|subtract|subtraction|mip|cad|ser|pe1?|segmentation|mask|voi)\b", text):
        return "derived_or_map", "derived_keyword"

    if re.search(r"\b(t2|stir|tirm|ssfse|frfse|tse\s*t2|fse\s*t2)\b", text):
        return "t2_like", "t2_keyword"

    if re.search(
        r"\b(t1|spgr|fspgr|ir[-_ ]?spgr|3d\s*f?gre|3dfgre|fgre|vibe|flash|tfl|thrive|dyn|dynamic|dce)\b",
        text,
    ):
        reasons.append("t1_keyword")

    if te is not None and te >= 50:
        return "t2_like", f"long_te:{te:g}"
    if tr is not None and te is not None and tr <= 1200 and te <= 30:
        reasons.append(f"short_tr_te:{tr:g}/{te:g}")

    if reasons:
        return "t1_like", ";".join(reasons)
    return "unknown", "no_t1_t2_clue"


def compact_counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value if value not in (None, "") else "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def unique_join(values: Iterable[str]) -> str:
    return " | ".join(sorted({value for value in values if value}))


def classify_exam(row: pd.Series) -> dict[str, object]:
    descriptions = split_pipe(row.get("selected_series_descriptions", ""))
    dirs = split_pipe(row.get("selected_series_dirs", ""))
    planes = []
    contrast_classes = []
    contrast_reasons = []
    tr_values = []
    te_values = []
    for index, series_dir_text in enumerate(dirs):
        description = descriptions[index] if index < len(descriptions) else Path(series_dir_text).name
        ds = read_first_header(str(Path(series_dir_text).resolve()))
        planes.append(infer_plane(description, ds))
        contrast_class, reason = infer_contrast_class(description, ds)
        contrast_classes.append(contrast_class)
        contrast_reasons.append(reason)
        values = numeric_header_values(ds)
        if values["tr"] is not None:
            tr_values.append(f"{values['tr']:g}")
        if values["te"] is not None:
            te_values.append(f"{values['te']:g}")

    nonempty_planes = [plane for plane in planes if plane]
    unique_planes = sorted(set(nonempty_planes))
    unique_contrasts = sorted(set(contrast_classes))
    if not dirs:
        exam_contrast = ""
    elif len(unique_contrasts) == 1:
        exam_contrast = unique_contrasts[0]
    else:
        exam_contrast = "mixed:" + "|".join(unique_contrasts)

    return {
        "source_series_count": len(dirs),
        "source_planes": " | ".join(planes),
        "source_plane_unique": " | ".join(unique_planes),
        "source_contrast_classes": " | ".join(contrast_classes),
        "source_contrast_unique": exam_contrast,
        "source_contrast_reasons": " | ".join(contrast_reasons),
        "source_tr_values": unique_join(tr_values),
        "source_te_values": unique_join(te_values),
    }


def matches_reference(observed: str, reference: str) -> object:
    observed_values = [value.strip() for value in observed.split("|") if value.strip()]
    reference_values = [value.strip() for value in reference.split("|") if value.strip()]
    if not observed_values or not reference_values:
        return ""
    return all(value in reference_values for value in observed_values)


def status_for_row(row: pd.Series) -> str:
    issues = []
    if not row["source_series_count"]:
        issues.append("no_selected_source_series")
    if row["nifti_header_view_matches_anchor"] is False:
        issues.append("nifti_header_view_mismatch_vs_anchor")
    if row["view_matches_anchor"] is False:
        issues.append("view_mismatch_vs_anchor")
    if row["view_matches_mamamia"] is False:
        issues.append("view_mismatch_vs_mamamia")
    contrast = str(row["source_contrast_unique"])
    if contrast == "unknown" or "unknown" in contrast:
        issues.append("contrast_unknown")
    if contrast.startswith("mixed:"):
        issues.append("mixed_contrast_selected_series")
    if row["contrast_matches_anchor"] is False:
        issues.append("contrast_mismatch_vs_anchor")
    return "ok" if not issues else "|".join(sorted(set(issues)))


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    audit_df = pd.read_csv(args.reconstruction_audit_csv)
    if not args.include_inactive:
        audit_df = audit_df[audit_df["reconstructed_exam_exists"].map(parse_bool)].copy()

    classified_rows = []
    for _, row in audit_df.iterrows():
        record = {
            "patient_id": row.get("patient_id", ""),
            "dataset": row.get("dataset", ""),
            "exam_date": row.get("exam_date", ""),
            "is_anchor_date": parse_bool(row.get("is_anchor_date", "")),
            "mamamia_view": str(row.get("mamamia_view", "") or "").strip().lower(),
            "anchor_reconstructed_date": row.get("anchor_reconstructed_date", ""),
            "reconstructed_exam_exists": parse_bool(row.get("reconstructed_exam_exists", "")),
            "reconstructed_exam_dir": row.get("reconstructed_exam_dir", ""),
            "reconstructed_patient_root_status": row.get("reconstructed_patient_root_status", ""),
            "selection_note": row.get("selection_note", ""),
            "selected_series_descriptions": row.get("selected_series_descriptions", ""),
            "selected_series_dirs": row.get("selected_series_dirs", ""),
            "risk_status_from_reconstruction_audit": row.get("risk_status", ""),
        }
        record.update(classify_exam(row))
        record.update(infer_view_from_nifti(first_reconstructed_phase(row.get("reconstructed_exam_dir", ""))))
        classified_rows.append(record)

    exam_df = pd.DataFrame(classified_rows)
    anchor_refs: dict[str, dict[str, str]] = {}
    for patient_id, patient_rows in exam_df.groupby("patient_id", sort=False):
        anchors = patient_rows[patient_rows["is_anchor_date"]]
        anchor_row = anchors.iloc[0] if not anchors.empty else patient_rows.iloc[0]
        anchor_plane = str(anchor_row.get("source_plane_unique", "") or "").strip()
        if not anchor_plane:
            anchor_plane = str(anchor_row.get("mamamia_view", "") or "").strip()
        anchor_nifti_view = str(anchor_row.get("nifti_header_view", "") or "").strip()
        anchor_contrast = str(anchor_row.get("source_contrast_unique", "") or "").strip()
        if not anchor_contrast or anchor_contrast == "unknown":
            anchor_contrast = "t1_like"
        anchor_refs[patient_id] = {
            "anchor_plane_reference": anchor_plane,
            "anchor_nifti_view_reference": anchor_nifti_view,
            "anchor_contrast_reference": anchor_contrast,
        }

    exam_df["anchor_plane_reference"] = exam_df["patient_id"].map(
        lambda patient_id: anchor_refs.get(patient_id, {}).get("anchor_plane_reference", "")
    )
    exam_df["anchor_contrast_reference"] = exam_df["patient_id"].map(
        lambda patient_id: anchor_refs.get(patient_id, {}).get("anchor_contrast_reference", "")
    )
    exam_df["anchor_nifti_view_reference"] = exam_df["patient_id"].map(
        lambda patient_id: anchor_refs.get(patient_id, {}).get("anchor_nifti_view_reference", "")
    )
    exam_df["view_matches_anchor"] = exam_df.apply(
        lambda row: matches_reference(row["source_plane_unique"], row["anchor_plane_reference"]),
        axis=1,
    )
    exam_df["nifti_header_view_matches_anchor"] = exam_df.apply(
        lambda row: matches_reference(row["nifti_header_view"], row["anchor_nifti_view_reference"]),
        axis=1,
    )
    exam_df["view_matches_mamamia"] = exam_df.apply(
        lambda row: matches_reference(row["source_plane_unique"], row["mamamia_view"]),
        axis=1,
    )
    exam_df["contrast_matches_anchor"] = exam_df.apply(
        lambda row: matches_reference(row["source_contrast_unique"], row["anchor_contrast_reference"]),
        axis=1,
    )
    exam_df["view_contrast_status"] = exam_df.apply(status_for_row, axis=1)

    patient_rows = []
    for patient_id, patient_df in exam_df.groupby("patient_id", sort=False):
        non_anchor = patient_df[~patient_df["is_anchor_date"]]
        mismatches = patient_df[patient_df["view_contrast_status"] != "ok"]
        patient_rows.append(
            {
                "patient_id": patient_id,
                "dataset": patient_df["dataset"].iloc[0],
                "num_active_exams": len(patient_df),
                "num_extra_exams": len(non_anchor),
                "anchor_plane_reference": patient_df["anchor_plane_reference"].iloc[0],
                "anchor_nifti_view_reference": patient_df["anchor_nifti_view_reference"].iloc[0],
                "anchor_contrast_reference": patient_df["anchor_contrast_reference"].iloc[0],
                "all_views_match_anchor": bool(
                    (patient_df["view_matches_anchor"] != False).all()
                ),
                "all_nifti_header_views_match_anchor": bool(
                    (patient_df["nifti_header_view_matches_anchor"] != False).all()
                ),
                "all_contrasts_match_anchor": bool(
                    (patient_df["contrast_matches_anchor"] != False).all()
                ),
                "mismatch_exam_dates": " | ".join(
                    str(value) for value in mismatches["exam_date"].tolist()
                ),
                "mismatch_statuses": " | ".join(
                    sorted(set(str(value) for value in mismatches["view_contrast_status"]))
                ),
            }
        )
    patient_df = pd.DataFrame(patient_rows)
    summary = {
        "reconstruction_audit_csv": str(args.reconstruction_audit_csv.resolve()),
        "include_inactive": bool(args.include_inactive),
        "exam_rows": int(len(exam_df)),
        "patient_rows": int(len(patient_df)),
        "status_counts": compact_counts(exam_df.get("view_contrast_status", [])),
        "source_plane_counts": compact_counts(exam_df.get("source_plane_unique", [])),
        "nifti_header_view_counts": compact_counts(exam_df.get("nifti_header_view", [])),
        "source_contrast_counts": compact_counts(exam_df.get("source_contrast_unique", [])),
        "view_matches_anchor_counts": compact_counts(exam_df.get("view_matches_anchor", [])),
        "nifti_header_view_matches_anchor_counts": compact_counts(
            exam_df.get("nifti_header_view_matches_anchor", [])
        ),
        "contrast_matches_anchor_counts": compact_counts(
            exam_df.get("contrast_matches_anchor", [])
        ),
        "patients_with_any_mismatch": int(
            (patient_df.get("mismatch_exam_dates", pd.Series(dtype=str)).astype(str) != "").sum()
        ),
    }
    return exam_df, patient_df, summary


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    exam_df, patient_df, summary = audit(args)

    exam_path = args.output_root / "view_contrast_exam_audit.csv"
    patient_path = args.output_root / "view_contrast_patient_summary.csv"
    summary_path = args.output_root / "summary.json"
    exam_df.to_csv(exam_path, index=False)
    patient_df.to_csv(patient_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Exam rows:    {summary['exam_rows']}")
    print(f"Patient rows: {summary['patient_rows']}")
    print("Status counts:")
    for key, value in summary["status_counts"].items():
        print(f"  {key}: {value}")
    print("Plane counts:")
    for key, value in summary["source_plane_counts"].items():
        print(f"  {key}: {value}")
    print("NIfTI header view counts:")
    for key, value in summary["nifti_header_view_counts"].items():
        print(f"  {key}: {value}")
    print("Contrast counts:")
    for key, value in summary["source_contrast_counts"].items():
        print(f"  {key}: {value}")
    print(f"Exam audit:   {exam_path}")
    print(f"Patient audit:{patient_path}")
    print(f"Summary:      {summary_path}")


if __name__ == "__main__":
    main()
