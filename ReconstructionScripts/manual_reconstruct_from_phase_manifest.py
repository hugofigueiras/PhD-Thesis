#!/usr/bin/env python3
"""Manually reconstruct NIfTI phases from an explicit DICOM source manifest.

This is for audited exception cases where one logical DCE phase is spread
across multiple DICOM series folders, such as paired superior/inferior slabs.
The script is intentionally explicit: the manifest defines which series folders
belong to each output phase.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import pydicom
import SimpleITK as sitk


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "Reconstructed_Datasets" / "ISPY1-Volumes"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct manual DCE phases from a phase-to-series CSV manifest."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually write NIfTI files. Without this flag the script only prints a dry run.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing volume_phase*.nii.gz files.",
    )
    return parser.parse_args()


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


def read_dicom(path: Path, *, pixels: bool) -> pydicom.Dataset:
    return pydicom.dcmread(str(path), stop_before_pixels=not pixels, force=True)


def parse_float_triplet(value: object) -> np.ndarray | None:
    if value is None or len(value) != 3:
        return None
    try:
        return np.array([float(item) for item in value], dtype=float)
    except Exception:
        return None


def parse_float_six(value: object) -> tuple[np.ndarray, np.ndarray] | None:
    if value is None or len(value) != 6:
        return None
    try:
        row_cos = np.array([float(item) for item in value[:3]], dtype=float)
        col_cos = np.array([float(item) for item in value[3:]], dtype=float)
    except Exception:
        return None
    return row_cos, col_cos


def phase_time(ds: pydicom.Dataset) -> str:
    value = getattr(ds, "AcquisitionTime", None)
    if value is None:
        value = getattr(ds, "ContentTime", "")
    return str(value)


def phase_summary(phase_files: Sequence[Path]) -> dict[str, object]:
    headers = [read_dicom(path, pixels=False) for path in phase_files]
    first = headers[0]
    iop = parse_float_six(getattr(first, "ImageOrientationPatient", None))
    slice_cos = np.array([0.0, 0.0, 1.0], dtype=float)
    if iop is not None:
        row_cos, col_cos = iop
        slice_cos = np.cross(row_cos, col_cos)

    projections = []
    times = []
    for ds in headers:
        ipp = parse_float_triplet(getattr(ds, "ImagePositionPatient", None))
        if ipp is not None:
            projections.append(float(np.dot(ipp, slice_cos)))
        times.append(phase_time(ds))

    spacing_z = ""
    if len(projections) >= 2:
        deltas = np.diff(sorted(projections))
        nonzero = [abs(delta) for delta in deltas if abs(delta) > 1e-8]
        if nonzero:
            spacing_z = f"{float(np.median(nonzero)):.6g}"

    pixel_spacing = getattr(first, "PixelSpacing", None)
    spacing_xy = ""
    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        spacing_xy = f"{float(pixel_spacing[0]):.6g}x{float(pixel_spacing[1]):.6g}"

    return {
        "num_slices": len(phase_files),
        "rows": getattr(first, "Rows", ""),
        "columns": getattr(first, "Columns", ""),
        "spacing_xy": spacing_xy,
        "spacing_z_from_positions": spacing_z,
        "first_time": min(times) if times else "",
        "last_time": max(times) if times else "",
        "min_position": round(min(projections), 6) if projections else "",
        "max_position": round(max(projections), 6) if projections else "",
    }


def build_sitk_image(phase_files: Sequence[Path]) -> sitk.Image:
    datasets = [read_dicom(path, pixels=True) for path in phase_files]
    if not datasets:
        raise ValueError("Cannot build a phase from zero DICOM files")

    first = datasets[0]
    iop = parse_float_six(getattr(first, "ImageOrientationPatient", None))
    if iop is None:
        row_cos = np.array([1.0, 0.0, 0.0], dtype=float)
        col_cos = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        row_cos, col_cos = iop
    slice_cos = np.cross(row_cos, col_cos)

    positions = [
        parse_float_triplet(getattr(ds, "ImagePositionPatient", None))
        for ds in datasets
    ]
    if all(position is not None for position in positions):
        projections = [float(np.dot(position, slice_cos)) for position in positions]
        order = sorted(range(len(datasets)), key=lambda index: projections[index])
        datasets = [datasets[index] for index in order]
        positions = [positions[index] for index in order]
    else:
        order = sorted(
            range(len(datasets)),
            key=lambda index: int(getattr(datasets[index], "InstanceNumber", index)),
        )
        datasets = [datasets[index] for index in order]
        positions = [positions[index] for index in order]

    slices = [ds.pixel_array for ds in datasets]
    volume = np.stack(slices, axis=0)
    image = sitk.GetImageFromArray(volume)

    pixel_spacing = getattr(first, "PixelSpacing", [1.0, 1.0])
    spacing_y = float(pixel_spacing[0])
    spacing_x = float(pixel_spacing[1])

    spacing_z = None
    valid_positions = [position for position in positions if position is not None]
    if len(valid_positions) >= 2 and len(valid_positions) == len(positions):
        projections = [float(np.dot(position, slice_cos)) for position in valid_positions]
        deltas = np.diff(projections)
        nonzero = [abs(delta) for delta in deltas if abs(delta) > 1e-8]
        if nonzero:
            spacing_z = float(np.median(nonzero))
            if float(np.median(deltas)) < 0:
                slice_cos = -slice_cos
    if spacing_z is None:
        for attr in ("SpacingBetweenSlices", "SliceThickness"):
            value = getattr(first, attr, None)
            if value is not None:
                spacing_z = float(value)
                break
    if spacing_z is None:
        spacing_z = 1.0

    origin_position = positions[0]
    origin = (
        tuple(float(item) for item in origin_position)
        if origin_position is not None
        else (0.0, 0.0, 0.0)
    )
    direction = (
        float(row_cos[0]), float(col_cos[0]), float(slice_cos[0]),
        float(row_cos[1]), float(col_cos[1]), float(slice_cos[1]),
        float(row_cos[2]), float(col_cos[2]), float(slice_cos[2]),
    )
    image.SetSpacing((spacing_x, spacing_y, spacing_z))
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def load_manifest(manifest_csv: Path) -> tuple[str, str, dict[int, list[Path]], list[dict[str, str]]]:
    with manifest_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_csv}")

    patient_ids = {row["patient_id"].strip() for row in rows}
    exam_dates = {row["exam_date"].strip() for row in rows}
    if len(patient_ids) != 1 or len(exam_dates) != 1:
        raise ValueError("Manifest must contain exactly one patient_id and one exam_date")

    grouped: dict[int, list[Path]] = defaultdict(list)
    for row in rows:
        phase_index = int(row["phase_index"])
        grouped[phase_index].append(Path(row["series_dir"]).expanduser())
    return patient_ids.pop(), exam_dates.pop(), dict(grouped), rows


def main() -> None:
    args = parse_args()
    patient_id, exam_date, phase_series_dirs, manifest_rows = load_manifest(args.manifest_csv)
    output_dir = args.output_root / patient_id / exam_date
    print(f"[manual reconstruction] patient={patient_id} exam_date={exam_date}")
    print(f"  manifest: {args.manifest_csv}")
    print(f"  output:   {output_dir}")
    print(f"  mode:     {'write' if args.write else 'dry_run'}")

    phase_file_groups: dict[int, list[Path]] = {}
    for phase_index, series_dirs in sorted(phase_series_dirs.items()):
        files: list[Path] = []
        for series_dir in series_dirs:
            series_files = list_dicom_files(series_dir)
            if not series_files:
                raise FileNotFoundError(f"No DICOM files found in {series_dir}")
            files.extend(series_files)
        phase_file_groups[phase_index] = files
        summary = phase_summary(files)
        print(
            "  phase "
            f"{phase_index}: series={len(series_dirs)} slices={summary['num_slices']} "
            f"matrix={summary['rows']}x{summary['columns']} "
            f"spacing={summary['spacing_xy']}x{summary['spacing_z_from_positions']} "
            f"time={summary['first_time']}..{summary['last_time']} "
            f"position={summary['min_position']}..{summary['max_position']}"
        )

    if not args.write:
        print("Dry run only; no files written.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for phase_index, files in sorted(phase_file_groups.items()):
        output_path = output_dir / f"volume_phase{phase_index}.nii.gz"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")
        image = build_sitk_image(files)
        sitk.WriteImage(image, str(output_path))
        print(f"  wrote: {output_path}")

    manifest_copy = output_dir / "manual_phase_source_manifest.csv"
    if manifest_copy.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {manifest_copy}")
    shutil.copy2(args.manifest_csv, manifest_copy)
    print(f"  copied manifest: {manifest_copy}")


if __name__ == "__main__":
    main()
