import argparse
import shutil
from pathlib import Path
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


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

    return phases


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


def count_mamamia_phases(images_root: Path, patient_id: str) -> Tuple[Optional[int], str]:
    """
    MAMA-MIA images are stored like:
      images_root / patient_id / patient_id_0000.nii.gz
      images_root / patient_id / patient_id_0001.nii.gz
    """
    patient_dir = images_root / patient_id

    if not patient_dir.exists() or not patient_dir.is_dir():
        return None, f"MAMA-MIA patient image directory not found: {patient_dir}"

    phase_files = sorted(patient_dir.glob(f"{patient_id}_*.nii.gz"))
    return len(phase_files), f"Found {len(phase_files)} MAMA-MIA image phases in {patient_dir}"


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
    Normalize sequence names for matching across time points.

    Removes only leading laterality-like prefixes, not trailing
    words like SER or PE1.

    Examples:
      'LEFT - Dynamic-3dfgre' -> 'dynamic-3dfgre'
      'RIGHT-Dynamic-3dfgre' -> 'dynamic-3dfgre'
      'Dynamic-3dfgre'       -> 'dynamic-3dfgre'
      'Dynamic-3dfgre SER'   -> 'dynamic-3dfgre ser'
      'Dynamic-3dfgre PE1'   -> 'dynamic-3dfgre pe1'
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

    This is stricter than substring matching and still keeps SER / PE1 separate
    because those suffixes remain after normalize_sequence_name().

    Examples:
      'Sagittal-IR_3DFGRE' -> 'dynamic-3dfgre'
      'Sagittal-IR3DFGRE'  -> 'dynamic-3dfgre'
      'Dynamic-3dfgre'     -> 'dynamic-3dfgre'
      'LEFT - Dynamic-3dfgre' -> 'dynamic-3dfgre'
      'Dynamic-3dfgre SER' -> 'dynamic-3dfgre ser'  (stays separate)
    """
    s = normalize_sequence_name(sequence_name)

    alias_map = {
        "sagittal-ir_3dfgre": "dynamic-3dfgre",
        "sagittal-ir3dfgre": "dynamic-3dfgre",
        "dynamic-3dfgre": "dynamic-3dfgre",
    }

    return alias_map.get(s, s)


def choose_best_series_for_exam(
    exam_dir: Path,
    anchor_sequence_name: str,
) -> Optional[Path]:
    """
    For non-anchor time points:
    - keep only folders whose normalized sequence family matches
      the anchor sequence family
    - among those, choose the one with the most DICOM files
    - if tied, choose lexicographically smallest folder name
    """
    candidates = []
    anchor_family = normalize_sequence_family(anchor_sequence_name)

    for series_dir in sorted(exam_dir.iterdir(), key=lambda p: p.name):
        if not series_dir.is_dir():
            continue

        candidate_sequence_name = extract_sequence_name(series_dir.name)
        candidate_family = normalize_sequence_family(candidate_sequence_name)

        if candidate_family != anchor_family:
            continue

        num_files = len(list_dicom_files_sorted(series_dir))
        candidates.append((num_files, series_dir))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (-x[0], x[1].name))
    return candidates[0][1]


def find_matching_series_in_other_dates(
    anchor_series_dir: Path,
    sequence_name: str,
) -> List[Tuple[str, Path]]:
    """
    Anchor exam date:
      use only the exact anchor_series_dir from metadata.csv

    Other exam dates:
      choose only one folder per date:
      the folder whose normalized sequence family matches
      and that has the most DICOM files
    """
    anchor_exam_dir = anchor_series_dir.parent
    patient_root = anchor_exam_dir.parent

    results = []

    for exam_dir in sorted(patient_root.iterdir(), key=lambda p: p.name):
        if not exam_dir.is_dir():
            continue

        exam_date_str = extract_exam_date_from_exam_folder(exam_dir.name)

        if exam_dir.resolve() == anchor_exam_dir.resolve():
            results.append((exam_date_str, anchor_series_dir))
            continue

        best_series = choose_best_series_for_exam(
            exam_dir=exam_dir,
            anchor_sequence_name=sequence_name,
        )

        if best_series is not None:
            results.append((exam_date_str, best_series))

    return results


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

            candidate_series = find_matching_series_in_other_dates(
                anchor_series_dir=anchor_series_dir,
                sequence_name=sequence_name,
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
                        f'No folders with matching sequence family were found. '
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
        description="Construct NIfTI volumes from matched DICOM folders, reconstruct corresponding sequence folders in other exam dates, and compare anchor-phase counts with MAMA-MIA."
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
        default=Path("data/Breast MRI NACT Pilot/manifest-RbPGRCVv7392292744865323559/metadata.csv"),
        help="Path to the metadata CSV file."
    )
    ap.add_argument(
        "--dataset_value",
        type=str,
        default="NACT",
        help='Dataset value to filter in the Excel file, e.g. "NACT".'
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("./NACT-Volumes"),
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
        default=200,
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