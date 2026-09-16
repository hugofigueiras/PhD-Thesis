import argparse
from pathlib import Path
import re
from typing import List, Optional

import numpy as np
import pydicom
import SimpleITK as sitk


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


def list_dicom_files_sorted(dicom_dir: Path) -> List[Path]:
    files = [p for p in dicom_dir.iterdir() if p.is_file()]
    files.sort(key=lambda p: p.name)
    return files


def split_into_phases(dicom_files: List[Path], phase_gap: int = 200) -> List[List[Path]]:
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


def extract_patient_id_from_series_dir(series_dir: Path) -> str:
    """
    Expected structure:
      ... / UCSF-BR-20 / 01-09-1992-... / 4.000000-Dynamic-3dfgre-07968

    Output:
      NACT_20
    """
    patient_folder = series_dir.parent.parent.name.strip()
    m = re.match(r"UCSF-BR-(\d+)", patient_folder, flags=re.IGNORECASE)
    if m:
        return f"NACT_{int(m.group(1)):02d}"
    return patient_folder


def extract_date_from_exam_dir(exam_dir: Path) -> str:
    """
    Example:
      01-09-1992-205329-MR BREAS UNIT-... -> 01-09-1992
    """
    m = re.match(r"^(\d{2}-\d{2}-\d{4})", exam_dir.name)
    if m:
        return m.group(1)
    return "unknown_date"


def load_series_dirs(series_dirs: List[Path], series_list: Optional[Path]) -> List[Path]:
    result = list(series_dirs)

    if series_list is not None:
        with open(series_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                result.append(Path(line))

    # deduplicate while preserving order
    seen = set()
    unique = []
    for p in result:
        rp = str(p.resolve()) if p.exists() else str(p)
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def reconstruct_series_dir(series_dir: Path, output_root: Path, phase_gap: int, dry_run: bool) -> None:
    if not series_dir.exists() or not series_dir.is_dir():
        print(f"[SKIP] Not a valid directory: {series_dir}")
        return

    exam_dir = series_dir.parent
    patient_id = extract_patient_id_from_series_dir(series_dir)
    date_str = extract_date_from_exam_dir(exam_dir)

    dicom_files = list_dicom_files_sorted(series_dir)
    if not dicom_files:
        print(f"[SKIP] No files found in: {series_dir}")
        return

    phases = split_into_phases(dicom_files, phase_gap=phase_gap)

    out_dir = output_root / patient_id / date_str

    print(f"[RECONSTRUCT]")
    print(f"  Series dir: {series_dir}")
    print(f"  Patient ID: {patient_id}")
    print(f"  Date: {date_str}")
    print(f"  Num DICOM files: {len(dicom_files)}")
    print(f"  Num phases: {len(phases)}")

    for phase_idx, phase_files in enumerate(phases):
        out_path = out_dir / f"volume_phase{phase_idx}.nii.gz"
        print(f"  -> phase {phase_idx}: {len(phase_files)} slices -> {out_path}")

        if dry_run:
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        img = build_sitk_image_from_phase_files(phase_files)
        sitk.WriteImage(img, str(out_path))


def main():
    ap = argparse.ArgumentParser(
        description="Manually reconstruct NIfTI volumes from explicitly chosen DICOM series folders."
    )
    ap.add_argument(
        "--series_dir",
        type=Path,
        action="append",
        default=[],
        help="One DICOM series folder to reconstruct. Can be provided multiple times."
    )
    ap.add_argument(
        "--series_list",
        type=Path,
        default=None,
        help="Text file with one DICOM series folder path per line."
    )
    ap.add_argument(
        "--output_root",
        type=Path,
        default=Path("./Manual-Volumes"),
        help="Root output folder."
    )
    ap.add_argument(
        "--phase_gap",
        type=int,
        default=10000,
        help="A new phase starts when AcquisitionTime or fallback ContentTime changes by at least this amount."
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, do not write anything; only print what would be done."
    )

    args = ap.parse_args()

    selected_series = load_series_dirs(args.series_dir, args.series_list)

    if not selected_series:
        raise SystemExit("No series folders were provided. Use --series_dir or --series_list.")

    for series_dir in selected_series:
        reconstruct_series_dir(
            series_dir=series_dir,
            output_root=args.output_root,
            phase_gap=args.phase_gap,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()