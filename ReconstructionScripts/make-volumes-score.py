import argparse
import shutil
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


# ----------------------------
# Basic helpers
# ----------------------------

def clean_uid(value) -> str:
    return str(value).strip()


def parse_time_to_number(value) -> Optional[int]:
    """
    DICOM time examples:
      '134512'
      '134512.123'
    Returns the integer part for comparison.
    """
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    s = re.sub(r"[^0-9.]", "", s)
    if not s:
        return None

    try:
        return int(float(s))
    except ValueError:
        return None


def get_phase_time_number(ds) -> Optional[int]:
    """
    Prefer AcquisitionTime (0008,0032).
    Fall back to ContentTime (0008,0033) if AcquisitionTime is missing.
    """
    acq_time = parse_time_to_number(getattr(ds, "AcquisitionTime", None))
    if acq_time is not None:
        return acq_time

    content_time = parse_time_to_number(getattr(ds, "ContentTime", None))
    if content_time is not None:
        return content_time

    return None


def get_phase_min_time(phase_files: List[Path]) -> Optional[int]:
    """
    Return the minimum available phase time among the files in a phase.
    Uses AcquisitionTime first, then ContentTime.
    """
    times = []

    for f in phase_files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            t = get_phase_time_number(ds)
            if t is not None:
                times.append(t)
        except Exception:
            continue

    if not times:
        return None

    return min(times)


def extract_exam_date_from_exam_folder(exam_folder_name: str) -> str:
    """
    Examples:
      03-29-1990-218019-MR BREAST-74237 -> 03-29-1990
      04-17-1990-148579-MR BREAST-06955 -> 04-17-1990
      07-03-1990-797021 -> 07-03-1990
    """
    m = re.match(r"^(\d{2}-\d{2}-\d{4})", exam_folder_name)
    if m:
        return m.group(1)
    return "unknown_date"


def list_dicom_files_sorted(dicom_dir: Path) -> List[Path]:
    files = [p for p in dicom_dir.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


# ----------------------------
# Phase splitting
# ----------------------------

def split_into_phases_by_acquisition_time(
    dicom_files: List[Path],
    phase_gap: int = 200
) -> List[List[Path]]:
    """
    Split a sorted list of DICOM files into phases.

    Rule:
    - Prefer AcquisitionTime
    - If missing, use ContentTime
    - Start a new phase when the chosen time changes by >= phase_gap
    - After splitting, reorder phases by their minimum available time so that:
        lowest time -> phase0
        next lowest -> phase1
        ...
    """
    if not dicom_files:
        return []

    phases = []
    current_phase = []
    prev_time = None

    for f in dicom_files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
        current_time = get_phase_time_number(ds)

        if not current_phase:
            current_phase.append(f)
            prev_time = current_time
            continue

        new_phase = False
        if prev_time is not None and current_time is not None:
            if abs(current_time - prev_time) >= phase_gap:
                new_phase = True

        if new_phase:
            phases.append(current_phase)
            current_phase = [f]
        else:
            current_phase.append(f)

        prev_time = current_time

    if current_phase:
        phases.append(current_phase)

    # Reorder phases by minimum available time
    decorated = []
    for idx, phase_files in enumerate(phases):
        min_time = get_phase_min_time(phase_files)
        # phases without valid time go to the end, preserving original order
        sort_key = (min_time is None, min_time if min_time is not None else idx)
        decorated.append((sort_key, phase_files))

    decorated.sort(key=lambda x: x[0])
    phases = [phase_files for _, phase_files in decorated]

    return phases


def count_detected_phases(series_dir: Path, phase_gap: int) -> Optional[int]:
    try:
        dicom_files = list_dicom_files_sorted(series_dir)
        if not dicom_files:
            return None
        phases = split_into_phases_by_acquisition_time(dicom_files, phase_gap=phase_gap)
        return len(phases)
    except Exception:
        return None


# ----------------------------
# DICOM metadata helpers
# ----------------------------

def get_first_dicom_dataset(series_dir: Path):
    dicom_files = list_dicom_files_sorted(series_dir)
    if not dicom_files:
        return None
    try:
        return pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return None


def get_rows_cols(ds) -> Tuple[Optional[int], Optional[int]]:
    rows = getattr(ds, "Rows", None)
    cols = getattr(ds, "Columns", None)
    return rows, cols


def get_pixel_spacing(ds) -> Optional[Tuple[float, float]]:
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None or len(ps) < 2:
        return None
    try:
        return float(ps[0]), float(ps[1])
    except Exception:
        return None


def get_orientation_label_from_text(text: str) -> Optional[str]:
    t = text.lower()
    if "sagitt" in t:
        return "sagittal"
    if "axial" in t or "transverse" in t:
        return "axial"
    if "coron" in t:
        return "coronal"
    return None


def get_orientation_label(ds, series_dir: Path) -> Optional[str]:
    # First try folder/description text
    seq = extract_sequence_name(series_dir.name)
    label = get_orientation_label_from_text(seq)
    if label is not None:
        return label

    # Then try SeriesDescription / ProtocolName
    for attr in ["SeriesDescription", "ProtocolName"]:
        val = getattr(ds, attr, None)
        if val is not None:
            label = get_orientation_label_from_text(str(val))
            if label is not None:
                return label

    return None


def get_text_fields(ds) -> str:
    vals = []
    for attr in [
        "SeriesDescription",
        "ProtocolName",
        "SequenceName",
        "ScanningSequence",
        "SequenceVariant",
        "ScanOptions",
        "MRAcquisitionType",
    ]:
        v = getattr(ds, attr, None)
        if v is not None:
            vals.append(str(v))
    return " | ".join(vals)


# ----------------------------
# Naming helpers
# ----------------------------

def extract_sequence_name(series_folder_name: str) -> str:
    """
    Examples:
      4.000000-Dynamic-3dfgre-05145 -> Dynamic-3dfgre
      41000.000000-Dynamic-3dfgre SER-79739 -> Dynamic-3dfgre SER
      41001.000000-Dynamic-3dfgre PE1-39347 -> Dynamic-3dfgre PE1
      4.000000-LEFT - Dynamic-3dfgre-05145 -> LEFT - Dynamic-3dfgre
    """
    parts = series_folder_name.split("-")
    if len(parts) < 3:
        return series_folder_name
    return "-".join(parts[1:-1]).strip()


def normalize_sequence_name(sequence_name: str) -> str:
    """
    Removes only leading laterality-like prefixes, not trailing
    words like SER or PE1.
    """
    s = sequence_name.strip()
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(
        r"^(left|right|lt|rt|bilateral)\s*[-: ]+\s*",
        "",
        s,
        flags=re.IGNORECASE,
    ).strip()
    s = re.sub(r"\s+", " ", s).strip()
    return s.lower()


def normalize_sequence_family(sequence_name: str) -> str:
    """
    Map known equivalent sequence descriptions into one family key.

    Keep this small and controlled.
    """
    s = normalize_sequence_name(sequence_name)

    alias_map = {
        "sagittal-ir_3dfgre": "dynamic-3dfgre",
        "sagittal-ir3dfgre": "dynamic-3dfgre",
        "dynamic-3dfgre": "dynamic-3dfgre",
    }

    return alias_map.get(s, s)


# ----------------------------
# Reconstruction
# ----------------------------

def build_sitk_image_from_phase_files(phase_files: List[Path]) -> sitk.Image:
    datasets = [pydicom.dcmread(str(f), force=True) for f in phase_files]

    slices = [ds.pixel_array for ds in datasets]
    volume = np.stack(slices, axis=0)

    img = sitk.GetImageFromArray(volume)

    first = datasets[0]

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    spacing_y = float(pixel_spacing[0])
    spacing_x = float(pixel_spacing[1])

    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        row_cos = np.array([float(x) for x in iop[:3]], dtype=float)
        col_cos = np.array([float(x) for x in iop[3:]], dtype=float)
        slice_cos = np.cross(row_cos, col_cos)
    else:
        row_cos = np.array([1.0, 0.0, 0.0], dtype=float)
        col_cos = np.array([0.0, 1.0, 0.0], dtype=float)
        slice_cos = np.array([0.0, 0.0, 1.0], dtype=float)

    positions = []
    for ds in datasets:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) == 3:
            positions.append(np.array([float(x) for x in ipp], dtype=float))
        else:
            positions.append(None)

    origin = tuple(float(x) for x in positions[0]) if positions[0] is not None else (0.0, 0.0, 0.0)

    spacing_z = None
    if len(positions) >= 2 and all(p is not None for p in positions):
        projected = [float(np.dot(p, slice_cos)) for p in positions]
        deltas = np.diff(projected)

        nonzero_deltas = [abs(d) for d in deltas if abs(d) > 1e-8]
        if nonzero_deltas:
            spacing_z = float(np.median(nonzero_deltas))

            signed_median = float(np.median(deltas))
            if signed_median < 0:
                slice_cos = -slice_cos

    if spacing_z is None:
        if hasattr(first, "SpacingBetweenSlices"):
            spacing_z = float(first.SpacingBetweenSlices)
        elif hasattr(first, "SliceThickness"):
            spacing_z = float(first.SliceThickness)
        else:
            spacing_z = 1.0

    direction = (
        float(row_cos[0]), float(col_cos[0]), float(slice_cos[0]),
        float(row_cos[1]), float(col_cos[1]), float(slice_cos[1]),
        float(row_cos[2]), float(col_cos[2]), float(slice_cos[2]),
    )

    img.SetSpacing((spacing_x, spacing_y, spacing_z))
    img.SetOrigin(origin)
    img.SetDirection(direction)

    return img


# ----------------------------
# MAMA-MIA phase count
# ----------------------------

def count_mamamia_phases(images_root: Path, patient_id: str) -> Tuple[Optional[int], str]:
    patient_dir = images_root / patient_id

    if not patient_dir.exists() or not patient_dir.is_dir():
        return None, f"MAMA-MIA patient image directory not found: {patient_dir}"

    phase_files = sorted(patient_dir.glob(f"{patient_id}_*.nii.gz"))
    return len(phase_files), f"Found {len(phase_files)} MAMA-MIA image phases in {patient_dir}"


# ----------------------------
# Report helpers
# ----------------------------

def add_report_row(
    report_rows,
    patient_id,
    acquisition_date,
    uid,
    dicom_dir,
    num_dicom_files,
    num_phases,
    phase_index,
    num_slices_in_phase,
    output_path,
    status,
    message,
    item_type="volume",
    exam_date_folder="",
    matched_sequence_name="",
    timepoint_role="",
    mamamia_num_phases="",
    target_num_phases="",
    same_num_phases="",
):
    report_rows.append({
        "patient_id": patient_id,
        "acquisition_date": acquisition_date,
        "tcia_series_uid": uid,
        "dicom_dir": str(dicom_dir) if dicom_dir is not None else "",
        "num_dicom_files": num_dicom_files,
        "num_phases": num_phases,
        "phase_index": phase_index,
        "num_slices_in_phase": num_slices_in_phase,
        "output_path": str(output_path) if output_path is not None else "",
        "item_type": item_type,
        "exam_date_folder": exam_date_folder,
        "matched_sequence_name": matched_sequence_name,
        "timepoint_role": timepoint_role,
        "mamamia_num_phases": mamamia_num_phases,
        "target_num_phases": target_num_phases,
        "same_num_phases": same_num_phases,
        "status": status,
        "message": message,
    })


# ----------------------------
# Candidate scoring for extra time points
# ----------------------------

def safe_abs_diff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    return abs(a - b)


def score_candidate_series(
    candidate_dir: Path,
    anchor_sequence_name: str,
    anchor_num_files: int,
    anchor_num_phases: Optional[int],
    anchor_ds,
    mamamia_num_phases: Optional[int],
    phase_gap: int,
) -> Tuple[float, str]:
    """
    Multi-factor score for extra time points.

    Main ideas:
    - orientation match is important
    - exact/family description match helps
    - file count closeness to anchor helps
    - phase count closeness to anchor/MAMA-MIA helps
    - matrix and spacing closeness help
    """
    reasons = []
    score = 0.0

    candidate_seq = extract_sequence_name(candidate_dir.name)
    anchor_norm = normalize_sequence_name(anchor_sequence_name)
    cand_norm = normalize_sequence_name(candidate_seq)
    anchor_family = normalize_sequence_family(anchor_sequence_name)
    cand_family = normalize_sequence_family(candidate_seq)

    # Description / family
    if cand_norm == anchor_norm:
        score += 6.0
        reasons.append("exact_normalized_name")
    elif cand_family == anchor_family:
        score += 4.0
        reasons.append("same_sequence_family")

    # Candidate metadata
    cand_ds = get_first_dicom_dataset(candidate_dir)
    cand_num_files = len(list_dicom_files_sorted(candidate_dir))
    cand_num_phases = count_detected_phases(candidate_dir, phase_gap=phase_gap)

    # Orientation / plane
    anchor_orient = get_orientation_label(anchor_ds, Path(f"0-0-{anchor_sequence_name}-0")) if anchor_ds is not None else get_orientation_label_from_text(anchor_sequence_name)
    cand_orient = get_orientation_label(cand_ds, candidate_dir) if cand_ds is not None else get_orientation_label_from_text(candidate_seq)

    if anchor_orient is not None and cand_orient is not None:
        if anchor_orient == cand_orient:
            score += 4.0
            reasons.append(f"same_orientation:{anchor_orient}")
        else:
            score -= 4.0
            reasons.append(f"different_orientation:{cand_orient}")

    # File count closeness to anchor
    diff_files = safe_abs_diff(cand_num_files, anchor_num_files)
    if diff_files is not None:
        if diff_files == 0:
            score += 4.0
            reasons.append("same_num_files")
        elif diff_files <= 5:
            score += 3.0
            reasons.append("very_close_num_files")
        elif diff_files <= 20:
            score += 1.5
            reasons.append("close_num_files")
        elif diff_files >= 80:
            score -= 2.0
            reasons.append("far_num_files")

        # Mild preference for common useful counts like 180, but not a rule
        if cand_num_files == 180:
            score += 0.5
            reasons.append("has_180_files")

    # Phase count closeness
    target_ref_phases = anchor_num_phases
    if target_ref_phases is None:
        target_ref_phases = mamamia_num_phases

    if cand_num_phases is not None and target_ref_phases is not None:
        if cand_num_phases == target_ref_phases:
            score += 3.5
            reasons.append("same_num_phases")
        else:
            phase_diff = abs(cand_num_phases - target_ref_phases)
            if phase_diff == 1:
                score += 1.0
                reasons.append("close_num_phases")
            else:
                score -= 1.5
                reasons.append("different_num_phases")

    # Matrix size closeness
    if anchor_ds is not None and cand_ds is not None:
        a_rows, a_cols = get_rows_cols(anchor_ds)
        c_rows, c_cols = get_rows_cols(cand_ds)

        if a_rows is not None and a_cols is not None and c_rows is not None and c_cols is not None:
            if a_rows == c_rows and a_cols == c_cols:
                score += 2.0
                reasons.append("same_matrix")
            elif abs(a_rows - c_rows) <= 16 and abs(a_cols - c_cols) <= 16:
                score += 0.5
                reasons.append("close_matrix")
            else:
                score -= 1.0
                reasons.append("different_matrix")

        # Pixel spacing closeness
        a_ps = get_pixel_spacing(anchor_ds)
        c_ps = get_pixel_spacing(cand_ds)
        if a_ps is not None and c_ps is not None:
            ps_diff = abs(a_ps[0] - c_ps[0]) + abs(a_ps[1] - c_ps[1])
            if ps_diff < 1e-4:
                score += 2.0
                reasons.append("same_spacing")
            elif ps_diff < 0.2:
                score += 0.5
                reasons.append("close_spacing")
            else:
                score -= 1.0
                reasons.append("different_spacing")

    return score, ";".join(reasons)


def choose_best_series_for_exam(
    exam_dir: Path,
    anchor_series_dir: Path,
    phase_gap: int,
    mamamia_num_phases: Optional[int],
) -> Tuple[Optional[Path], str]:
    """
    For non-anchor time points:
    - score all series in that exam
    - choose the highest-scoring one
    - if top score is weak or almost tied, mark ambiguous
    """
    anchor_sequence_name = extract_sequence_name(anchor_series_dir.name)
    anchor_num_files = len(list_dicom_files_sorted(anchor_series_dir))
    anchor_num_phases = count_detected_phases(anchor_series_dir, phase_gap=phase_gap)
    anchor_ds = get_first_dicom_dataset(anchor_series_dir)

    scored = []

    for series_dir in sorted(exam_dir.iterdir(), key=lambda p: p.name):
        if not series_dir.is_dir():
            continue

        score, reasons = score_candidate_series(
            candidate_dir=series_dir,
            anchor_sequence_name=anchor_sequence_name,
            anchor_num_files=anchor_num_files,
            anchor_num_phases=anchor_num_phases,
            anchor_ds=anchor_ds,
            mamamia_num_phases=mamamia_num_phases,
            phase_gap=phase_gap,
        )
        scored.append((score, reasons, series_dir))

    if not scored:
        return None, "no_series_dirs"

    scored.sort(key=lambda x: (-x[0], x[2].name))
    best_score, best_reasons, best_dir = scored[0]

    # Ambiguity handling
    if len(scored) > 1:
        second_score = scored[1][0]
        if best_score < 4.0:
            return None, f"ambiguous_low_confidence: best={best_dir.name} score={best_score:.2f} reasons={best_reasons}"
        if abs(best_score - second_score) < 1.0:
            return None, (
                f"ambiguous_close_scores: best={best_dir.name} score={best_score:.2f} reasons={best_reasons}; "
                f"second={scored[1][2].name} score={second_score:.2f}"
            )

    if best_score < 4.0:
        return None, f"low_confidence: best={best_dir.name} score={best_score:.2f} reasons={best_reasons}"

    return best_dir, f"selected: score={best_score:.2f}; reasons={best_reasons}"


def find_matching_series_in_other_dates(
    anchor_series_dir: Path,
    phase_gap: int,
    mamamia_num_phases: Optional[int],
) -> Tuple[List[Tuple[str, Path]], List[Tuple[str, str]]]:
    """
    Anchor exam date:
      use only the exact anchor_series_dir from metadata.csv

    Other exam dates:
      choose only one folder per date using scoring
    """
    anchor_exam_dir = anchor_series_dir.parent
    patient_root = anchor_exam_dir.parent

    results = []
    notes = []

    for exam_dir in sorted(patient_root.iterdir(), key=lambda p: p.name):
        if not exam_dir.is_dir():
            continue

        exam_date_str = extract_exam_date_from_exam_folder(exam_dir.name)

        if exam_dir.resolve() == anchor_exam_dir.resolve():
            results.append((exam_date_str, anchor_series_dir))
            continue

        best_series, note = choose_best_series_for_exam(
            exam_dir=exam_dir,
            anchor_series_dir=anchor_series_dir,
            phase_gap=phase_gap,
            mamamia_num_phases=mamamia_num_phases,
        )

        notes.append((exam_date_str, note))

        if best_series is not None:
            results.append((exam_date_str, best_series))

    return results, notes


# ----------------------------
# Write volumes
# ----------------------------

def reconstruct_one_series_dir(
    patient_id: str,
    acquisition_date: str,
    uid: str,
    series_dir: Path,
    output_root: Path,
    phase_gap: int,
    dry_run: bool,
    report_rows,
    matched_sequence_name: str,
    timepoint_role: str,
    mamamia_num_phases=None,
):
    dicom_files = list_dicom_files_sorted(series_dir)

    if not dicom_files:
        print(f"[EMPTY DIR] patient={patient_id}")
        print(f"  Series dir: {series_dir}")
        add_report_row(
            report_rows=report_rows,
            patient_id=patient_id,
            acquisition_date=acquisition_date,
            uid=uid,
            dicom_dir=series_dir,
            num_dicom_files=0,
            num_phases=0,
            phase_index="",
            num_slices_in_phase=0,
            output_path=None,
            status="empty_dicom_dir",
            message="Directory exists but contains no files",
            item_type="volume",
            exam_date_folder=series_dir.parent.name,
            matched_sequence_name=matched_sequence_name,
            timepoint_role=timepoint_role,
            mamamia_num_phases=mamamia_num_phases if timepoint_role == "anchor_timepoint" else "",
            target_num_phases=0 if timepoint_role == "anchor_timepoint" else "",
            same_num_phases=(
                mamamia_num_phases == 0
                if (timepoint_role == "anchor_timepoint" and mamamia_num_phases is not None)
                else ""
            ),
        )
        return 0

    try:
        phases = split_into_phases_by_acquisition_time(dicom_files, phase_gap=phase_gap)
    except Exception as e:
        print(f"[PHASE SPLIT ERROR] patient={patient_id}: {e}")
        add_report_row(
            report_rows=report_rows,
            patient_id=patient_id,
            acquisition_date=acquisition_date,
            uid=uid,
            dicom_dir=series_dir,
            num_dicom_files=len(dicom_files),
            num_phases=0,
            phase_index="",
            num_slices_in_phase=0,
            output_path=None,
            status="phase_split_error",
            message=str(e),
            item_type="volume",
            exam_date_folder=series_dir.parent.name,
            matched_sequence_name=matched_sequence_name,
            timepoint_role=timepoint_role,
            mamamia_num_phases=mamamia_num_phases if timepoint_role == "anchor_timepoint" else "",
            target_num_phases=0 if timepoint_role == "anchor_timepoint" else "",
            same_num_phases="",
        )
        return 0

    print(f"[RECONSTRUCT] patient={patient_id} date={acquisition_date}")
    print(f"  Series dir: {series_dir}")
    print(f"  Files: {len(dicom_files)}")
    print(f"  Phases found: {len(phases)}")

    if timepoint_role == "anchor_timepoint" and mamamia_num_phases is not None:
        same_num_phases = (mamamia_num_phases == len(phases))
        print(f"  MAMA-MIA phases: {mamamia_num_phases}")
        print(f"  Same number of phases: {same_num_phases}")
    else:
        same_num_phases = ""

    patient_out_dir = output_root / patient_id / acquisition_date
    written_count = 0

    for phase_idx, phase_files in enumerate(phases):
        out_path = patient_out_dir / f"volume_phase{phase_idx}.nii.gz"
        print(f"  -> phase {phase_idx}: {len(phase_files)} slices -> {out_path}")

        if dry_run:
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=series_dir,
                num_dicom_files=len(dicom_files),
                num_phases=len(phases),
                phase_index=phase_idx,
                num_slices_in_phase=len(phase_files),
                output_path=out_path,
                status="dry_run",
                message="Dry run enabled; volume not written",
                item_type="volume",
                exam_date_folder=series_dir.parent.name,
                matched_sequence_name=matched_sequence_name,
                timepoint_role=timepoint_role,
                mamamia_num_phases=mamamia_num_phases if timepoint_role == "anchor_timepoint" else "",
                target_num_phases=len(phases) if timepoint_role == "anchor_timepoint" else "",
                same_num_phases=same_num_phases,
            )
            continue

        try:
            patient_out_dir.mkdir(parents=True, exist_ok=True)
            img = build_sitk_image_from_phase_files(phase_files)
            sitk.WriteImage(img, str(out_path))
            written_count += 1

            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=series_dir,
                num_dicom_files=len(dicom_files),
                num_phases=len(phases),
                phase_index=phase_idx,
                num_slices_in_phase=len(phase_files),
                output_path=out_path,
                status="written",
                message="NIfTI volume written successfully",
                item_type="volume",
                exam_date_folder=series_dir.parent.name,
                matched_sequence_name=matched_sequence_name,
                timepoint_role=timepoint_role,
                mamamia_num_phases=mamamia_num_phases if timepoint_role == "anchor_timepoint" else "",
                target_num_phases=len(phases) if timepoint_role == "anchor_timepoint" else "",
                same_num_phases=same_num_phases,
            )
        except Exception as e:
            print(f"  [WRITE ERROR] phase {phase_idx}: {e}")
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=series_dir,
                num_dicom_files=len(dicom_files),
                num_phases=len(phases),
                phase_index=phase_idx,
                num_slices_in_phase=len(phase_files),
                output_path=out_path,
                status="write_error",
                message=str(e),
                item_type="volume",
                exam_date_folder=series_dir.parent.name,
                matched_sequence_name=matched_sequence_name,
                timepoint_role=timepoint_role,
                mamamia_num_phases=mamamia_num_phases if timepoint_role == "anchor_timepoint" else "",
                target_num_phases=len(phases) if timepoint_role == "anchor_timepoint" else "",
                same_num_phases=same_num_phases,
            )

    return written_count


# ----------------------------
# Masks
# ----------------------------

def copy_patient_masks(
    patient_id: str,
    patient_out_dir: Path,
    seg_auto_root: Path,
    seg_expert_root: Path,
    report_rows,
    acquisition_date: str,
    uid: str,
    dry_run: bool,
) -> None:
    auto_src = seg_auto_root / f"{patient_id}.nii.gz"
    expert_src = seg_expert_root / f"{patient_id}.nii.gz"

    auto_dst = patient_out_dir / f"{patient_id}_automatic.nii.gz"
    expert_dst = patient_out_dir / f"{patient_id}_expert.nii.gz"

    for src, dst, label in [
        (auto_src, auto_dst, "automatic_mask"),
        (expert_src, expert_dst, "expert_mask"),
    ]:
        if not src.exists():
            print(f"  [MASK MISSING] {label}: {src}")
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=None,
                num_dicom_files=0,
                num_phases=0,
                phase_index="",
                num_slices_in_phase=0,
                output_path=dst,
                status="mask_missing",
                message=f"Source mask not found: {src}",
                item_type=label,
                timepoint_role="anchor_timepoint",
            )
            continue

        print(f"  [MASK] {src} -> {dst}")

        if dry_run:
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=None,
                num_dicom_files=0,
                num_phases=0,
                phase_index="",
                num_slices_in_phase=0,
                output_path=dst,
                status="dry_run",
                message=f"Dry run enabled; {label} not copied",
                item_type=label,
                timepoint_role="anchor_timepoint",
            )
            continue

        try:
            patient_out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=None,
                num_dicom_files=0,
                num_phases=0,
                phase_index="",
                num_slices_in_phase=0,
                output_path=dst,
                status="copied",
                message=f"{label} copied successfully",
                item_type=label,
                timepoint_role="anchor_timepoint",
            )
        except Exception as e:
            print(f"  [MASK COPY ERROR] {label}: {e}")
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date=acquisition_date,
                uid=uid,
                dicom_dir=None,
                num_dicom_files=0,
                num_phases=0,
                phase_index="",
                num_slices_in_phase=0,
                output_path=dst,
                status="mask_copy_error",
                message=str(e),
                item_type=label,
                timepoint_role="anchor_timepoint",
            )


def exam_dir_is_anchor(series_dir: Path, anchor_series_dir: Path) -> bool:
    return series_dir.resolve() == anchor_series_dir.resolve()


# ----------------------------
# Main processing
# ----------------------------

def process_dataset(
    excel_path: Path,
    csv_path: Path,
    dataset_value: str,
    output_root: Path,
    report_csv: Path,
    seg_auto_root: Path,
    seg_expert_root: Path,
    mamamia_images_root: Path,
    phase_gap: int,
    dry_run: bool,
) -> None:
    excel_df = pd.read_excel(excel_path)
    csv_df = pd.read_csv(csv_path)

    excel_df["dataset"] = excel_df["dataset"].astype(str).str.strip()
    excel_df["tcia_series_uid"] = excel_df["tcia_series_uid"].astype(str).str.strip()
    excel_df["patient_id"] = excel_df["patient_id"].astype(str).str.strip()

    csv_df["Series UID"] = csv_df["Series UID"].astype(str).str.strip()
    csv_df["File Location"] = csv_df["File Location"].astype(str).str.strip()

    filtered_excel = excel_df[excel_df["dataset"] == dataset_value.strip()]
    report_rows = []

    if filtered_excel.empty:
        print(f'No rows found in Excel with dataset == "{dataset_value}".')
        pd.DataFrame(report_rows).to_csv(report_csv, index=False)
        print(f"CSV report written to: {report_csv}")
        return

    total_matches = 0
    total_written = 0
    copied_masks_for_patient_date = set()
    reconstructed_series_dirs = set()

    for _, row in filtered_excel.iterrows():
        patient_id = row["patient_id"]
        uid = clean_uid(row["tcia_series_uid"])

        matches = csv_df[csv_df["Series UID"] == uid]

        if matches.empty:
            print(f"[NOT FOUND] patient={patient_id} uid={uid}")
            add_report_row(
                report_rows=report_rows,
                patient_id=patient_id,
                acquisition_date="unknown_date",
                uid=uid,
                dicom_dir=None,
                num_dicom_files=0,
                num_phases=0,
                phase_index="",
                num_slices_in_phase=0,
                output_path=None,
                status="uid_not_found_in_metadata",
                message="No matching Series UID found in metadata.csv",
                item_type="volume",
                timepoint_role="anchor_timepoint",
            )
            continue

        for _, match_row in matches.iterrows():
            total_matches += 1

            file_location = str(match_row["File Location"]).strip()
            anchor_series_dir = (csv_path.parent / file_location).resolve()

            if not anchor_series_dir.exists() or not anchor_series_dir.is_dir():
                print(f"[MISSING DIR] patient={patient_id} uid={uid}")
                print(f"  Series dir: {anchor_series_dir}")
                add_report_row(
                    report_rows=report_rows,
                    patient_id=patient_id,
                    acquisition_date="unknown_date",
                    uid=uid,
                    dicom_dir=anchor_series_dir,
                    num_dicom_files=0,
                    num_phases=0,
                    phase_index="",
                    num_slices_in_phase=0,
                    output_path=None,
                    status="missing_dicom_dir",
                    message="Matched File Location does not exist or is not a directory",
                    item_type="volume",
                    exam_date_folder=anchor_series_dir.parent.name if anchor_series_dir.parent else "",
                    matched_sequence_name="",
                    timepoint_role="anchor_timepoint",
                )
                continue

            anchor_output_date = extract_exam_date_from_exam_folder(anchor_series_dir.parent.name)

            mamamia_num_phases, mamamia_phase_message = count_mamamia_phases(
                images_root=mamamia_images_root,
                patient_id=patient_id,
            )
            print(f"[MAMA-MIA PHASE CHECK] {mamamia_phase_message}")

            sequence_name = extract_sequence_name(anchor_series_dir.name)
            sequence_name_norm = normalize_sequence_name(sequence_name)
            sequence_family = normalize_sequence_family(sequence_name)

            print(f"[ANCHOR] patient={patient_id} uid={uid}")
            print(f"  Anchor series: {anchor_series_dir.name}")
            print(f"  Anchor exam folder: {anchor_series_dir.parent.name}")
            print(f"  Anchor output date: {anchor_output_date}")
            print(f"  Raw sequence name: {sequence_name}")
            print(f"  Normalized sequence name: {sequence_name_norm}")
            print(f"  Sequence family: {sequence_family}")

            candidate_series, candidate_notes = find_matching_series_in_other_dates(
                anchor_series_dir=anchor_series_dir,
                phase_gap=phase_gap,
                mamamia_num_phases=mamamia_num_phases,
            )

            for exam_date_str, note in candidate_notes:
                print(f"  [MATCH NOTE] {exam_date_str}: {note}")
                if note.startswith("ambiguous") or note.startswith("low_confidence") or note.startswith("no_series_dirs"):
                    add_report_row(
                        report_rows=report_rows,
                        patient_id=patient_id,
                        acquisition_date=exam_date_str,
                        uid=uid,
                        dicom_dir=None,
                        num_dicom_files=0,
                        num_phases=0,
                        phase_index="",
                        num_slices_in_phase=0,
                        output_path=None,
                        status="candidate_selection_note",
                        message=note,
                        item_type="volume",
                        exam_date_folder=exam_date_str,
                        matched_sequence_name=sequence_name,
                        timepoint_role="extra_timepoint",
                        mamamia_num_phases="",
                        target_num_phases="",
                        same_num_phases="",
                    )

            if not candidate_series:
                print("  No matching series found in any exam date.")
                add_report_row(
                    report_rows=report_rows,
                    patient_id=patient_id,
                    acquisition_date=anchor_output_date,
                    uid=uid,
                    dicom_dir=anchor_series_dir,
                    num_dicom_files=0,
                    num_phases=0,
                    phase_index="",
                    num_slices_in_phase=0,
                    output_path=None,
                    status="no_matching_sequence_in_other_dates",
                    message=(
                        f'No folders with a sufficiently strong score were found. '
                        f'Raw anchor sequence="{sequence_name}", '
                        f'normalized="{sequence_name_norm}", '
                        f'family="{sequence_family}"'
                    ),
                    item_type="volume",
                    exam_date_folder=anchor_series_dir.parent.name,
                    matched_sequence_name=sequence_name,
                    timepoint_role="anchor_timepoint",
                    mamamia_num_phases=mamamia_num_phases if mamamia_num_phases is not None else "",
                    target_num_phases="",
                    same_num_phases="",
                )
                continue

            for exam_date_str, series_dir in candidate_series:
                key = str(series_dir.resolve())
                if key in reconstructed_series_dirs:
                    continue

                reconstructed_series_dirs.add(key)

                role = "anchor_timepoint" if exam_dir_is_anchor(series_dir, anchor_series_dir) else "extra_timepoint"

                total_written += reconstruct_one_series_dir(
                    patient_id=patient_id,
                    acquisition_date=exam_date_str,
                    uid=uid,
                    series_dir=series_dir,
                    output_root=output_root,
                    phase_gap=phase_gap,
                    dry_run=dry_run,
                    report_rows=report_rows,
                    matched_sequence_name=sequence_name,
                    timepoint_role=role,
                    mamamia_num_phases=mamamia_num_phases if role == "anchor_timepoint" else None,
                )

            mask_key = (patient_id, anchor_output_date)
            if mask_key not in copied_masks_for_patient_date:
                copy_patient_masks(
                    patient_id=patient_id,
                    patient_out_dir=output_root / patient_id / anchor_output_date,
                    seg_auto_root=seg_auto_root,
                    seg_expert_root=seg_expert_root,
                    report_rows=report_rows,
                    acquisition_date=anchor_output_date,
                    uid=uid,
                    dry_run=dry_run,
                )
                copied_masks_for_patient_date.add(mask_key)

    report_df = pd.DataFrame(report_rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print("\nDone.")
    print(f"Matched anchor series: {total_matches}")
    if dry_run:
        print("Dry run enabled: no files were written or copied.")
    else:
        print(f"Written NIfTI volumes: {total_written}")
    print(f"CSV report written to: {report_csv}")


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Construct NIfTI volumes from matched DICOM folders, reconstruct corresponding "
            "sequence folders in other exam dates using scoring, and compare anchor-phase "
            "counts with MAMA-MIA."
        )
    )
    ap.add_argument(
        "--excel_path",
        type=Path,
        default=Path("data/MAMA-MIA/clinical_and_imaging_info.xlsx"),
        help="Path to the Excel file."
    )
    ap.add_argument(
        "--csv_path",
        type=Path,
        default=Path("data/ISPY-2/manifest-1641168072464/metadata.csv"),
        help="Path to the metadata CSV file."
    )
    ap.add_argument(
        "--dataset_value",
        type=str,
        default="ISPY2",
        help='Dataset value to filter in the Excel file, e.g. "NACT".'
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("Reconstructed_Datasets/ISPY2-Volumes"),
        help="Root output folder where patient/date/volume_phaseX.nii.gz will be created."
    )
    ap.add_argument(
        "--report_csv",
        type=Path,
        default=Path("nifti_construction_report.csv"),
        help="Path to the CSV report."
    )
    ap.add_argument(
        "--seg_auto_root",
        type=Path,
        default=Path("data/MAMA-MIA/segmentations/automatic"),
        help="Folder containing automatic masks from MAMA-MIA."
    )
    ap.add_argument(
        "--seg_expert_root",
        type=Path,
        default=Path("data/MAMA-MIA/segmentations/expert"),
        help="Folder containing expert masks from MAMA-MIA."
    )
    ap.add_argument(
        "--mamamia_images_root",
        type=Path,
        default=Path("data/MAMA-MIA/images"),
        help="Root folder containing MAMA-MIA image phases stored as patient_id/patient_id_0000.nii.gz, etc."
    )
    ap.add_argument(
        "--phase_gap",
        type=int,
        default=80,
        help="A new phase starts when AcquisitionTime or fallback ContentTime changes by at least this amount."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not write or copy anything; only print what would be done."
    )

    args = ap.parse_args()

    process_dataset(
        excel_path=args.excel_path,
        csv_path=args.csv_path,
        dataset_value=args.dataset_value,
        output_root=args.output_root,
        report_csv=args.report_csv,
        seg_auto_root=args.seg_auto_root,
        seg_expert_root=args.seg_expert_root,
        mamamia_images_root=args.mamamia_images_root,
        phase_gap=args.phase_gap,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()