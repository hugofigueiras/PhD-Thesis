#!/usr/bin/env python3
"""Audit reconstructed longitudinal breast MRI volumes against source data.

This script is intentionally read-only for the dataset. It checks whether the
current reconstructed NIfTI tree is consistent with:

1. the MAMA-MIA anchor exam metadata and phase files, and
2. the original TCIA metadata/DICOM headers used to choose extra time points.

The audit is conservative: it flags anything that needs manual review rather
than silently deciding that a questionable series is correct.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd
import pydicom


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_RECONSTRUCTED_ROOT = PROJECT_ROOT / "Reconstructed_Datasets"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "reconstruction_consistency_audit"
)
DEFAULT_MANUAL_PHASE_MANIFEST_DIR = PROJECT_ROOT / "ReconstructionScripts" / "manual_overrides"


@dataclass(frozen=True)
class DatasetConfig:
    dataset: str
    reconstructed_dir: str
    metadata_csv: Path
    metadata_base: Path
    phase_gap: int


DATASET_CONFIGS = {
    "ISPY1": DatasetConfig(
        dataset="ISPY1",
        reconstructed_dir="ISPY1-Volumes",
        metadata_csv=Path(
            "data/ISPY-1/manifest-PyHQgfru6393647793776378748/metadata.csv"
        ),
        metadata_base=Path("data/ISPY-1/manifest-PyHQgfru6393647793776378748"),
        phase_gap=200,
    ),
    "ISPY2": DatasetConfig(
        dataset="ISPY2",
        reconstructed_dir="ISPY2-Volumes",
        metadata_csv=Path("data/ISPY-2/manifest-1641168072464/metadata.csv"),
        metadata_base=Path("data/ISPY-2/manifest-1641168072464"),
        phase_gap=80,
    ),
    "NACT": DatasetConfig(
        dataset="NACT",
        reconstructed_dir="NACT-Volumes",
        metadata_csv=Path(
            "data/Breast MRI NACT Pilot/"
            "manifest-RbPGRCVv7392292744865323559/metadata.csv"
        ),
        metadata_base=Path(
            "data/Breast MRI NACT Pilot/"
            "manifest-RbPGRCVv7392292744865323559"
        ),
        phase_gap=200,
    ),
}

SERIES_PHASE_STOPWORDS = {
    "aligned",
    "contrast",
    "cropped",
    "inject",
    "injection",
    "original",
    "registered",
    "repeat",
    "test",
    "to",
    "uni",
    "unilateral",
    "with",
}

PHASE_ONLY_DYNAMIC_TOKEN = "phase_only_dynamic"


@dataclass
class SeriesCandidate:
    dataset: str
    subject_id: str
    exam_key: str
    exam_dir: Path
    study_date: str
    study_description: str
    series_uid: str
    series_dir: Path
    series_description: str
    modality: str
    num_images: int | None
    family_key: str
    family_tokens: set[str]


@dataclass
class SelectedExam:
    exam_date: str
    exam_dir: Path | None
    selected_candidates: list[SeriesCandidate]
    selection_note: str
    selection_score: float | None
    phase_source_kind: str
    phases: list[list[Path]]
    candidate_rows: list[dict[str, object]]


@dataclass
class ManualPhaseManifest:
    path: Path
    patient_id: str
    exam_date: str
    phase_dirs: dict[int, list[Path]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit reconstructed MAMA-MIA longitudinal volumes against original "
            "DICOM metadata, DICOM headers, and MAMA-MIA anchor phase files."
        )
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--reconstructed-root", type=Path, default=DEFAULT_RECONSTRUCTED_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--manual-phase-manifest-dir",
        type=Path,
        default=DEFAULT_MANUAL_PHASE_MANIFEST_DIR,
        help=(
            "Folder containing explicit patient/date phase-source CSV manifests for "
            "manual exception cases."
        ),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=sorted(DATASET_CONFIGS),
        choices=sorted(DATASET_CONFIGS),
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional MAMA-MIA patient IDs to audit, e.g. ISPY1_1001 ISPY2_100899.",
    )
    parser.add_argument(
        "--max-patients-per-dataset",
        type=int,
        default=None,
        help="Limit patients per dataset for smoke checks.",
    )
    parser.add_argument(
        "--timepoints",
        choices=("all", "anchor", "extra"),
        default="all",
        help="Audit all reconstructed dates, only MAMA-MIA anchor dates, or only extra dates.",
    )
    parser.add_argument(
        "--include-metadata-only-exams",
        action="store_true",
        help=(
            "Also audit original metadata exams that are not present in the reconstructed "
            "tree. By default the audit focuses on reconstructed dates plus the anchor date."
        ),
    )
    parser.add_argument(
        "--deep-dicom-phase-check",
        action="store_true",
        help="Read DICOM headers from selected series to re-split phases and compare counts.",
    )
    parser.add_argument(
        "--candidate-phase-counts",
        action="store_true",
        help=(
            "Also estimate phase counts for every candidate series in candidate_ranking.csv. "
            "This is useful but slower on a full run."
        ),
    )
    parser.add_argument(
        "--candidate-audit",
        choices=("selected", "all", "none"),
        default="selected",
        help=(
            "How many source-series candidate rows to write. 'selected' is the fast "
            "default; 'all' writes full alternative rankings for manual review."
        ),
    )
    parser.add_argument(
        "--compare-anchor-pixels",
        action="store_true",
        help=(
            "Load reconstructed and MAMA-MIA anchor NIfTI arrays and compute downsampled "
            "similarity under identity/flip/swap-XY transforms."
        ),
    )
    parser.add_argument(
        "--check-duplicate-reconstructed-volumes",
        action="store_true",
        help=(
            "Load reconstructed NIfTI arrays and flag exact duplicate phase volumes across "
            "exam dates for the same patient. Useful for targeted QC; slower on a full run."
        ),
    )
    parser.add_argument(
        "--max-anchor-pixel-comparisons",
        type=int,
        default=None,
        help="Optional cap on anchor image comparisons.",
    )
    parser.add_argument(
        "--similarity-stride",
        type=int,
        nargs=3,
        default=(4, 4, 2),
        metavar=("X", "Y", "Z"),
        help="Voxel stride for fast image-similarity comparisons.",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-3)
    parser.add_argument("--spacing-atol", type=float, default=1e-3)
    parser.add_argument("--min-ispy2-family-similarity", type=float, default=0.45)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N patients. Use 0 to disable.",
    )
    return parser.parse_args()


def compact_counts(values: Iterable[object]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value if value not in (None, "") else "missing")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def load_manual_phase_manifests(manifest_dir: Path) -> dict[tuple[str, str], ManualPhaseManifest]:
    manifests: dict[tuple[str, str], ManualPhaseManifest] = {}
    if not manifest_dir.exists():
        return manifests
    required = {"patient_id", "exam_date", "phase_index", "series_dir"}
    for path in sorted(manifest_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = set(reader.fieldnames or [])
            if not required.issubset(fieldnames):
                continue
            rows = list(reader)
        if not rows:
            continue
        patient_ids = {row["patient_id"].strip() for row in rows}
        exam_dates = {row["exam_date"].strip() for row in rows}
        if len(patient_ids) != 1 or len(exam_dates) != 1:
            continue
        phase_dirs: dict[int, list[Path]] = defaultdict(list)
        for row in rows:
            try:
                phase_index = int(row["phase_index"])
            except Exception:
                continue
            series_dir = Path(row["series_dir"]).expanduser().resolve()
            phase_dirs[phase_index].append(series_dir)
        if phase_dirs:
            patient_id = patient_ids.pop()
            exam_date = exam_dates.pop()
            manifests[(patient_id, exam_date)] = ManualPhaseManifest(
                path=path,
                patient_id=patient_id,
                exam_date=exam_date,
                phase_dirs=dict(phase_dirs),
            )
    return manifests


def clean_uid(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def date_to_folder(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return text.replace("/", "-")
    return parsed.strftime("%m-%d-%Y")


def normalize_ispy2_patient_id(value: object) -> str:
    return str(value).strip().upper().replace("_", "-")


def metadata_subject_id(patient_id: str, dataset: str) -> str:
    if dataset == "ISPY2":
        return normalize_ispy2_patient_id(patient_id)
    if dataset == "NACT":
        match = re.search(r"(\d+)$", patient_id)
        if match:
            return f"UCSF-BR-{int(match.group(1)):02d}"
    return str(patient_id).strip()


def reconstructed_patient_id(patient_id: str, dataset: str) -> str:
    if dataset == "ISPY2":
        return normalize_ispy2_patient_id(patient_id)
    return str(patient_id).strip()


def resolve_metadata_location(config: DatasetConfig, value: object) -> Path:
    path = Path(str(value).strip())
    if path.is_absolute():
        return path
    text = str(path)
    if text.startswith("./"):
        text = text[2:]
        path = Path(text)
    return (config.metadata_base / path).resolve()


def parse_int(value: object) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(float(str(value).strip()))
    except Exception:
        return None


def parse_time_to_number(value: object) -> int | None:
    if value is None:
        return None
    text = re.sub(r"[^0-9.]", "", str(value).strip())
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def get_phase_time_number(ds: pydicom.Dataset) -> int | None:
    for attr in ("AcquisitionTime", "ContentTime"):
        value = parse_time_to_number(getattr(ds, attr, None))
        if value is not None:
            return value
    return None


def list_dicom_files_sorted(series_dir: Path) -> list[Path]:
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
    files.sort(key=lambda path: path.name)
    return files


@lru_cache(maxsize=20000)
def read_first_dicom_header(series_dir_text: str) -> pydicom.Dataset | None:
    files = list_dicom_files_sorted(Path(series_dir_text))
    if not files:
        return None
    try:
        return pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return None


def read_dicom_header(path: Path) -> pydicom.Dataset | None:
    try:
        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def first_dicom_header(series_dir: Path) -> pydicom.Dataset | None:
    return read_first_dicom_header(str(series_dir.resolve()))


def dicom_text_fields(ds: pydicom.Dataset | None) -> str:
    if ds is None:
        return ""
    values: list[str] = []
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
            values.append(str(value))
    return " | ".join(values)


def get_orientation_label_from_text(text: str) -> str:
    normalized = text.lower()
    if "sagitt" in normalized:
        return "sagittal"
    if "axial" in normalized or "transverse" in normalized:
        return "axial"
    if "coron" in normalized:
        return "coronal"
    return ""


def plane_from_iop(value: object) -> str:
    try:
        if value is None or len(value) != 6:
            return ""
        row_cos = np.array([float(x) for x in value[:3]], dtype=float)
        col_cos = np.array([float(x) for x in value[3:]], dtype=float)
        normal = np.cross(row_cos, col_cos)
        axis = int(np.argmax(np.abs(normal)))
    except Exception:
        return ""
    if axis == 0:
        return "sagittal"
    if axis == 1:
        return "coronal"
    return "axial"


def get_orientation_label(ds: pydicom.Dataset | None, text: str) -> str:
    label = get_orientation_label_from_text(text)
    if label:
        return label
    if ds is not None:
        label = get_orientation_label_from_text(dicom_text_fields(ds))
        if label:
            return label
        return plane_from_iop(getattr(ds, "ImageOrientationPatient", None))
    return ""


def extract_series_description_from_folder(series_folder_name: str) -> str:
    parts = series_folder_name.split("-")
    if len(parts) < 3:
        return series_folder_name
    return "-".join(parts[1:-1]).strip()


def is_phase_only_dynamic_series(text: str) -> bool:
    normalized = str(text).strip()
    normalized = re.sub(r"ispy\s*2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"ispy2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    if not normalized:
        return False
    tokens = normalized.split()
    return bool(tokens) and all(
        re.fullmatch(r"(pre|post\d*|ph\d+|phase\d+)", token) for token in tokens
    )


def normalize_series_name(text: str) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", normalized)
    normalized = normalized.replace(":", " ")
    normalized = re.sub(r"ispy\s*2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"ispy2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"ph\s*\d+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"phase\s*\d+", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"pre\s*--\s*registered", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"pre[-\s]*contrast", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"post[-\s]*contrast", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"(prepost|delayed|delay|post|pre|registered|repeat|test)",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"(\d+(?:\.\d+)?)\s*ml", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def normalize_sequence_family(text: str) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(
        r"^(left|right|lt|rt|bilateral)\s*[-: ]+\s*",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip()
    normalized = re.sub(r"\s+", " ", normalized).strip().lower()
    aliases = {
        "sagittal-ir_3dfgre": "dynamic-3dfgre",
        "sagittal-ir3dfgre": "dynamic-3dfgre",
        "sagittal-ir3dfgre pe1": "dynamic-3dfgre pe1",
        "sagittal-ir3dfgre ser": "dynamic-3dfgre ser",
        "dynamic-3dfgre": "dynamic-3dfgre",
    }
    return aliases.get(normalized, normalized)


def tokenize_series_family(text: str) -> set[str]:
    normalized = normalize_series_name(text)
    tokens = []
    for token in normalized.split():
        if not token or token in SERIES_PHASE_STOPWORDS or token.isdigit():
            continue
        tokens.append(token)
    if not tokens and is_phase_only_dynamic_series(text):
        return {PHASE_ONLY_DYNAMIC_TOKEN}
    return set(tokens)


def series_family_key(text: str) -> str:
    return " ".join(sorted(tokenize_series_family(text)))


def family_similarity(anchor_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not anchor_tokens or not candidate_tokens:
        return 0.0
    overlap = len(anchor_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0
    dice = (2.0 * overlap) / (len(anchor_tokens) + len(candidate_tokens))
    containment = overlap / len(anchor_tokens)
    return max(dice, containment)


def series_phase_number(series_description: str) -> int | None:
    match = re.search(r"ph\s*(\d+)", series_description, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"post\s*(\d+)", series_description, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    normalized = series_description.lower()
    if (
        " pre " in f" {normalized} "
        or normalized.endswith(" pre")
        or "pre -- registered" in normalized
    ):
        return 0
    return None


def phase_min_time(phase_files: Sequence[Path]) -> int | None:
    values: list[int] = []
    for path in phase_files:
        ds = read_dicom_header(path)
        if ds is None:
            continue
        value = get_phase_time_number(ds)
        if value is not None:
            values.append(value)
    if not values:
        return None
    return min(values)


def series_min_time(series_dir: Path) -> int | None:
    return phase_min_time(list_dicom_files_sorted(series_dir))


def split_into_phases_by_acquisition_time(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> list[list[Path]]:
    if not dicom_files:
        return []
    phases: list[list[Path]] = []
    current_phase: list[Path] = []
    previous_time: int | None = None
    for path in dicom_files:
        ds = read_dicom_header(path)
        current_time = get_phase_time_number(ds) if ds is not None else None
        if not current_phase:
            current_phase.append(path)
            previous_time = current_time
            continue
        start_new_phase = False
        if previous_time is not None and current_time is not None:
            start_new_phase = abs(current_time - previous_time) >= phase_gap
        if start_new_phase:
            phases.append(current_phase)
            current_phase = [path]
        else:
            current_phase.append(path)
        previous_time = current_time
    if current_phase:
        phases.append(current_phase)
    return phases


def split_into_phases_by_grouped_time(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> list[list[Path]]:
    indexed_times: list[tuple[int, Path, int]] = []
    for index, path in enumerate(dicom_files):
        ds = read_dicom_header(path)
        if ds is None:
            return []
        current_time = get_phase_time_number(ds)
        if current_time is None:
            return []
        indexed_times.append((index, path, current_time))
    if not indexed_times:
        return []
    unique_times = sorted({time_value for _, _, time_value in indexed_times})
    time_to_cluster: dict[int, int] = {}
    cluster_to_time: dict[int, int] = {}
    cluster_index = -1
    previous_time = None
    for time_value in unique_times:
        if previous_time is None or abs(time_value - previous_time) >= phase_gap:
            cluster_index += 1
            cluster_to_time[cluster_index] = time_value
        time_to_cluster[time_value] = cluster_index
        previous_time = time_value

    grouped: dict[int, list[tuple[int, Path]]] = defaultdict(list)
    for index, path, time_value in indexed_times:
        grouped[time_to_cluster[time_value]].append((index, path))

    decorated = []
    for cluster_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        decorated.append((cluster_to_time.get(cluster_id, cluster_id), [path for _, path in items]))
    decorated.sort(key=lambda item: item[0])
    return [phase_files for _, phase_files in decorated]


def split_into_phases_by_position_reset(
    dicom_files: Sequence[Path],
    tolerance: float = 1e-3,
) -> list[list[Path]]:
    projections: list[float] = []
    slice_cos = None
    for path in dicom_files:
        ds = read_dicom_header(path)
        if ds is None:
            return []
        iop = getattr(ds, "ImageOrientationPatient", None)
        ipp = getattr(ds, "ImagePositionPatient", None)
        if iop is None or ipp is None or len(iop) != 6 or len(ipp) != 3:
            return []
        if slice_cos is None:
            row_cos = np.array([float(x) for x in iop[:3]], dtype=float)
            col_cos = np.array([float(x) for x in iop[3:]], dtype=float)
            slice_cos = np.cross(row_cos, col_cos)
        position = np.array([float(x) for x in ipp], dtype=float)
        projections.append(float(np.dot(position, slice_cos)))
    if not projections:
        return []
    deltas = [
        curr - prev
        for prev, curr in zip(projections[:-1], projections[1:])
        if abs(curr - prev) > tolerance
    ]
    if not deltas:
        return []
    direction = 1.0 if float(np.median(deltas)) >= 0 else -1.0
    boundaries = [0]
    for index in range(1, len(projections)):
        previous_projection = direction * projections[index - 1]
        current_projection = direction * projections[index]
        if current_projection < previous_projection - tolerance:
            boundaries.append(index)
    if len(boundaries) == 1:
        return []
    boundaries.append(len(dicom_files))
    phases = [
        list(dicom_files[start:end])
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    sizes = [len(phase_files) for phase_files in phases]
    if min(sizes) < 2 or max(sizes) - min(sizes) > 2:
        return []
    return phases


def phase_size_balance(phases: Sequence[Sequence[Path]]) -> float:
    if not phases:
        return 0.0
    sizes = [len(phase_files) for phase_files in phases]
    return min(sizes) / max(sizes) if max(sizes) > 0 else 0.0


def is_plausible_phase_split(phases: Sequence[Sequence[Path]]) -> bool:
    if not phases:
        return False
    sizes = [len(phase_files) for phase_files in phases]
    if min(sizes) < 2:
        return False
    return len(phases) == 1 or phase_size_balance(phases) >= 0.5


def split_single_series_into_phases(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> tuple[list[list[Path]], str]:
    candidates = [
        ([list(dicom_files)] if dicom_files else [], "single_series_as_is", 0),
        (
            split_into_phases_by_grouped_time(dicom_files, phase_gap=phase_gap),
            "single_series_split_by_grouped_time",
            3,
        ),
        (
            split_into_phases_by_position_reset(dicom_files),
            "single_series_split_by_position_reset",
            2,
        ),
        (
            split_into_phases_by_acquisition_time(dicom_files, phase_gap=phase_gap),
            "single_series_split_by_consecutive_time",
            1,
        ),
    ]
    plausible = [
        (phases, source_kind, priority)
        for phases, source_kind, priority in candidates
        if is_plausible_phase_split(phases)
    ]
    usable = plausible if plausible else candidates
    phases, source_kind, _ = max(
        usable,
        key=lambda item: (len(item[0]), phase_size_balance(item[0]), item[2]),
    )
    return phases, source_kind


def dicom_spacing_and_shape(phase_files: Sequence[Path]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not phase_files:
        return (), ()
    headers = [read_dicom_header(path) for path in phase_files]
    headers = [header for header in headers if header is not None]
    if not headers:
        return (), ()
    first = headers[0]
    rows = parse_int(getattr(first, "Rows", None))
    cols = parse_int(getattr(first, "Columns", None))
    shape = tuple(int(v) for v in (cols, rows, len(phase_files)) if v is not None)
    pixel_spacing = getattr(first, "PixelSpacing", None)
    spacing_x = spacing_y = None
    try:
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            spacing_y = float(pixel_spacing[0])
            spacing_x = float(pixel_spacing[1])
    except Exception:
        spacing_x = spacing_y = None

    spacing_z = None
    iop = getattr(first, "ImageOrientationPatient", None)
    if iop is not None and len(iop) == 6:
        try:
            row_cos = np.array([float(x) for x in iop[:3]], dtype=float)
            col_cos = np.array([float(x) for x in iop[3:]], dtype=float)
            slice_cos = np.cross(row_cos, col_cos)
            positions = []
            for ds in headers:
                ipp = getattr(ds, "ImagePositionPatient", None)
                if ipp is None or len(ipp) != 3:
                    positions = []
                    break
                positions.append(np.array([float(x) for x in ipp], dtype=float))
            if len(positions) >= 2:
                projected = [float(np.dot(position, slice_cos)) for position in positions]
                deltas = np.diff(projected)
                nonzero = [abs(delta) for delta in deltas if abs(delta) > 1e-8]
                if nonzero:
                    spacing_z = float(np.median(nonzero))
        except Exception:
            spacing_z = None
    if spacing_z is None:
        for attr in ("SpacingBetweenSlices", "SliceThickness"):
            try:
                value = getattr(first, attr, None)
                if value is not None:
                    spacing_z = float(value)
                    break
            except Exception:
                pass
    spacing = tuple(
        float(v) for v in (spacing_x, spacing_y, spacing_z) if v is not None
    )
    return shape, spacing


def load_metadata(config: DatasetConfig) -> tuple[dict[str, list[SeriesCandidate]], dict[str, SeriesCandidate]]:
    metadata = pd.read_csv(config.metadata_csv, dtype=str)
    required = {
        "Series UID",
        "Subject ID",
        "Study Description",
        "Study Date",
        "Series Description",
        "Modality",
        "Number of Images",
        "File Location",
    }
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"{config.metadata_csv} missing required columns: {missing}")

    by_subject: dict[str, list[SeriesCandidate]] = defaultdict(list)
    by_uid: dict[str, SeriesCandidate] = {}
    for _, row in metadata.iterrows():
        series_description = str(row["Series Description"]).strip()
        candidate = SeriesCandidate(
            dataset=config.dataset,
            subject_id=str(row["Subject ID"]).strip(),
            exam_key=str(resolve_metadata_location(config, row["File Location"]).parent),
            exam_dir=resolve_metadata_location(config, row["File Location"]).parent,
            study_date=date_to_folder(row["Study Date"]),
            study_description=str(row["Study Description"]).strip(),
            series_uid=str(row["Series UID"]).strip(),
            series_dir=resolve_metadata_location(config, row["File Location"]),
            series_description=series_description,
            modality=str(row["Modality"]).strip(),
            num_images=parse_int(row["Number of Images"]),
            family_key=series_family_key(series_description),
            family_tokens=tokenize_series_family(series_description),
        )
        by_subject[candidate.subject_id].append(candidate)
        by_uid[candidate.series_uid] = candidate
    return by_subject, by_uid


def group_by_exam(candidates: Sequence[SeriesCandidate]) -> dict[str, list[SeriesCandidate]]:
    grouped: dict[str, list[SeriesCandidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate.study_date].append(candidate)
    return grouped


def shape_from_dicom_header(ds: pydicom.Dataset | None) -> str:
    if ds is None:
        return ""
    rows = parse_int(getattr(ds, "Rows", None))
    cols = parse_int(getattr(ds, "Columns", None))
    if rows is None or cols is None:
        return ""
    return f"{rows}x{cols}"


def spacing_from_dicom_header(ds: pydicom.Dataset | None) -> str:
    if ds is None:
        return ""
    pixel_spacing = getattr(ds, "PixelSpacing", None)
    values: list[str] = []
    try:
        if pixel_spacing is not None and len(pixel_spacing) >= 2:
            values.extend([f"{float(pixel_spacing[0]):.6g}", f"{float(pixel_spacing[1]):.6g}"])
    except Exception:
        pass
    try:
        thickness = getattr(ds, "SliceThickness", None)
        if thickness is not None:
            values.append(f"{float(thickness):.6g}")
    except Exception:
        pass
    return "x".join(values)


def score_ispy2_candidate(
    candidate: SeriesCandidate,
    anchor: SeriesCandidate,
    compute_phase_count: bool,
    phase_gap: int,
) -> tuple[float, str, int | None]:
    reasons: list[str] = []
    score = 0.0
    candidate_tokens = candidate.family_tokens
    anchor_tokens = anchor.family_tokens
    similarity = family_similarity(anchor_tokens, candidate_tokens)
    if candidate.family_key == anchor.family_key:
        score += 1.0
        reasons.append("same_family_key")
    if similarity > 0:
        score += similarity
        reasons.append(f"token_similarity:{similarity:.3f}")
    if candidate.series_uid == anchor.series_uid:
        score += 2.0
        reasons.append("anchor_uid")

    anchor_ds = first_dicom_header(anchor.series_dir)
    candidate_ds = first_dicom_header(candidate.series_dir)

    anchor_plane = get_orientation_label(anchor_ds, anchor.series_description)
    candidate_plane = get_orientation_label(candidate_ds, candidate.series_description)
    if anchor_plane and candidate_plane:
        if anchor_plane == candidate_plane:
            score += 0.7
            reasons.append(f"same_plane:{candidate_plane}")
        else:
            score -= 1.0
            reasons.append(f"different_plane:{candidate_plane}")

    if anchor.num_images is not None and candidate.num_images is not None:
        diff = abs(anchor.num_images - candidate.num_images)
        if diff == 0:
            score += 0.6
            reasons.append("same_num_images")
        elif diff <= 10:
            score += 0.3
            reasons.append("close_num_images")
        elif diff >= 80:
            score -= 0.3
            reasons.append("far_num_images")

    anchor_shape = shape_from_dicom_header(anchor_ds)
    candidate_shape = shape_from_dicom_header(candidate_ds)
    if anchor_shape and candidate_shape:
        if anchor_shape == candidate_shape:
            score += 0.5
            reasons.append("same_matrix")
        else:
            score -= 0.2
            reasons.append("different_matrix")

    anchor_spacing = spacing_from_dicom_header(anchor_ds)
    candidate_spacing = spacing_from_dicom_header(candidate_ds)
    if anchor_spacing and candidate_spacing:
        if anchor_spacing == candidate_spacing:
            score += 0.5
            reasons.append("same_spacing")
        else:
            reasons.append("different_spacing")

    phase_count = None
    if compute_phase_count:
        phases, _ = split_single_series_into_phases(
            list_dicom_files_sorted(candidate.series_dir),
            phase_gap=phase_gap,
        )
        phase_count = len(phases)
    return score, ";".join(reasons), phase_count


def ispy1_nact_candidate_row(
    candidate: SeriesCandidate,
    exam_date: str,
    config: DatasetConfig,
    anchor_family: str,
    selected: bool,
    compute_phase_count: bool,
) -> dict[str, object]:
    ds = first_dicom_header(candidate.series_dir)
    candidate_family = normalize_sequence_family(candidate.series_description)
    candidate_phase_count = None
    if compute_phase_count:
        phases = split_into_phases_by_acquisition_time(
            list_dicom_files_sorted(candidate.series_dir),
            phase_gap=config.phase_gap,
        )
        candidate_phase_count = len(phases)
    return {
        "dataset": config.dataset,
        "exam_date": exam_date,
        "series_uid": candidate.series_uid,
        "series_description": candidate.series_description,
        "series_dir": str(candidate.series_dir),
        "modality": candidate.modality,
        "num_images_metadata": candidate.num_images,
        "candidate_family_key": candidate_family,
        "anchor_family_key": anchor_family,
        "plane": get_orientation_label(ds, candidate.series_description),
        "matrix": shape_from_dicom_header(ds),
        "spacing": spacing_from_dicom_header(ds),
        "candidate_phase_count": candidate_phase_count,
        "selection_score": "",
        "selection_reasons": "family_match" if selected else "",
        "selected": selected,
    }


def select_ispy1_or_nact_exam(
    exam_candidates: Sequence[SeriesCandidate],
    anchor: SeriesCandidate,
    exam_date: str,
    config: DatasetConfig,
    deep_phase_check: bool,
    candidate_phase_counts: bool,
    candidate_audit: str,
) -> SelectedExam:
    anchor_family = normalize_sequence_family(anchor.series_description)
    rows = []
    eligible = []
    for candidate in sorted(exam_candidates, key=lambda item: item.series_dir.name):
        candidate_family = normalize_sequence_family(candidate.series_description)
        selected_candidate = (
            candidate.modality == "MR"
            and candidate.series_dir.exists()
            and candidate_family == anchor_family
        )
        if selected_candidate:
            eligible.append(candidate)
        if candidate_audit == "all":
            rows.append(
                ispy1_nact_candidate_row(
                    candidate,
                    exam_date,
                    config,
                    anchor_family,
                    selected=False,
                    compute_phase_count=candidate_phase_counts,
                )
            )

    if not eligible:
        return SelectedExam(
            exam_date=exam_date,
            exam_dir=exam_candidates[0].exam_dir if exam_candidates else None,
            selected_candidates=[],
            selection_note=f"no_series_matching_anchor_family:{anchor_family}",
            selection_score=None,
            phase_source_kind="",
            phases=[],
            candidate_rows=rows,
        )

    eligible.sort(
        key=lambda item: (
            -(item.num_images if item.num_images is not None else len(list_dicom_files_sorted(item.series_dir))),
            item.series_dir.name,
        )
    )
    selected = eligible[0]
    if candidate_audit == "all":
        for row in rows:
            row["selected"] = row["series_uid"] == selected.series_uid
            if row["selected"]:
                row["selection_reasons"] = "family_match"
    elif candidate_audit == "selected":
        rows.append(
            ispy1_nact_candidate_row(
                selected,
                exam_date,
                config,
                anchor_family,
                selected=True,
                compute_phase_count=candidate_phase_counts,
            )
        )

    phases: list[list[Path]] = []
    phase_source_kind = "not_checked"
    if deep_phase_check:
        phases = split_into_phases_by_acquisition_time(
            list_dicom_files_sorted(selected.series_dir),
            phase_gap=config.phase_gap,
        )
        phase_source_kind = "single_series_split_by_consecutive_time"
    return SelectedExam(
        exam_date=exam_date,
        exam_dir=selected.exam_dir,
        selected_candidates=[selected],
        selection_note=f"selected_same_family_max_images:{anchor_family}",
        selection_score=None,
        phase_source_kind=phase_source_kind,
        phases=phases,
        candidate_rows=rows,
    )


def select_ispy2_exam(
    exam_candidates: Sequence[SeriesCandidate],
    anchor: SeriesCandidate,
    exam_date: str,
    config: DatasetConfig,
    min_similarity: float,
    deep_phase_check: bool,
    candidate_phase_counts: bool,
    candidate_audit: str,
) -> SelectedExam:
    usable = [
        candidate
        for candidate in exam_candidates
        if candidate.modality == "MR" and candidate.series_dir.exists()
    ]
    if not usable:
        return SelectedExam(
            exam_date=exam_date,
            exam_dir=exam_candidates[0].exam_dir if exam_candidates else None,
            selected_candidates=[],
            selection_note="no_usable_mr_series",
            selection_score=0.0,
            phase_source_kind="",
            phases=[],
            candidate_rows=[],
        )

    groups: dict[str, list[SeriesCandidate]] = defaultdict(list)
    for candidate in usable:
        groups[candidate.family_key or candidate.series_description.lower()].append(candidate)

    anchor_key = anchor.family_key or anchor.series_description.lower()
    group_scores = []
    candidate_rows = []
    for group_key, candidates in groups.items():
        group_tokens: set[str] = set()
        for candidate in candidates:
            group_tokens.update(candidate.family_tokens)
        score = family_similarity(anchor.family_tokens, group_tokens)
        if group_key == anchor_key:
            score += 1.0
        if any(candidate.series_uid == anchor.series_uid for candidate in candidates):
            score += 2.0
        group_scores.append((score, group_key, candidates))

    group_scores.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_group_key, best_candidates = group_scores[0]
    exact_anchor_group = groups.get(anchor_key)
    if exact_anchor_group is not None and best_score < 1.0:
        best_group_key = anchor_key
        best_candidates = exact_anchor_group
        best_score = max(best_score, 1.0)

    if best_score < min_similarity:
        best_candidates = []
        note = f"no_family_match_above_threshold:{best_score:.3f}; best={best_group_key}"
    else:
        note = (
            f"selected_family={best_group_key}; candidates={len(best_candidates)}; "
            f"score={best_score:.3f}"
        )
        if len(group_scores) > 1:
            second_score = group_scores[1][0]
            if abs(best_score - second_score) < 0.05:
                note += f"; second_best_score={second_score:.3f}"

    ordered = sort_candidates_by_phase(best_candidates, use_dicom_time=deep_phase_check)
    selected_uids = {candidate.series_uid for candidate in ordered}
    candidate_rank: dict[str, tuple[int, float]] = {}
    for rank, (group_score, _group_key, candidates) in enumerate(group_scores, start=1):
        for candidate in candidates:
            candidate_rank[candidate.series_uid] = (rank, group_score)

    if candidate_audit == "all":
        audit_candidates = [
            candidate
            for _rank, _group_score, candidates in group_scores
            for candidate in sorted(candidates, key=lambda item: item.series_dir.name)
        ]
    elif candidate_audit == "selected":
        audit_candidates = ordered
    else:
        audit_candidates = []

    for candidate in audit_candidates:
        rank, group_score = candidate_rank.get(candidate.series_uid, ("", ""))
        score, reasons, candidate_phase_count = score_ispy2_candidate(
            candidate,
            anchor,
            compute_phase_count=candidate_phase_counts,
            phase_gap=config.phase_gap,
        )
        ds = first_dicom_header(candidate.series_dir)
        candidate_rows.append(
            {
                "dataset": config.dataset,
                "exam_date": exam_date,
                "series_uid": candidate.series_uid,
                "series_description": candidate.series_description,
                "series_dir": str(candidate.series_dir),
                "modality": candidate.modality,
                "num_images_metadata": candidate.num_images,
                "candidate_family_key": candidate.family_key,
                "anchor_family_key": anchor_key,
                "plane": get_orientation_label(ds, candidate.series_description),
                "matrix": shape_from_dicom_header(ds),
                "spacing": spacing_from_dicom_header(ds),
                "candidate_phase_count": candidate_phase_count,
                "selection_group_rank": rank,
                "selection_group_score": group_score,
                "selection_score": score,
                "selection_reasons": reasons,
                "selected": candidate.series_uid in selected_uids,
            }
        )

    phases: list[list[Path]] = []
    phase_source_kind = "not_checked"
    if deep_phase_check and ordered:
        if len(ordered) == 1:
            phases, phase_source_kind = split_single_series_into_phases(
                list_dicom_files_sorted(ordered[0].series_dir),
                phase_gap=config.phase_gap,
            )
        else:
            phase_source_kind = "multi_series_group"
            source_kinds: list[str] = []
            any_internal_split = False
            for candidate in ordered:
                candidate_phases, source_kind = split_single_series_into_phases(
                    list_dicom_files_sorted(candidate.series_dir),
                    phase_gap=config.phase_gap,
                )
                if len(candidate_phases) > 1:
                    any_internal_split = True
                source_kinds.append(source_kind)
                phases.extend(candidate_phases)
            if any_internal_split:
                phase_source_kind = "multi_series_group_with_internal_splits:" + ",".join(
                    sorted(set(source_kinds))
                )
    return SelectedExam(
        exam_date=exam_date,
        exam_dir=ordered[0].exam_dir if ordered else None,
        selected_candidates=ordered,
        selection_note=note,
        selection_score=best_score,
        phase_source_kind=phase_source_kind,
        phases=phases,
        candidate_rows=candidate_rows,
    )


def sort_candidates_by_phase(
    candidates: Sequence[SeriesCandidate],
    use_dicom_time: bool,
) -> list[SeriesCandidate]:
    decorated = []
    for candidate in candidates:
        min_time = series_min_time(candidate.series_dir) if use_dicom_time else None
        phase_number = series_phase_number(candidate.series_description)
        sort_key = (
            min_time is None,
            min_time if min_time is not None else 0,
            phase_number is None,
            phase_number if phase_number is not None else 0,
            candidate.series_dir.name,
        )
        decorated.append((sort_key, candidate))
    decorated.sort(key=lambda item: item[0])
    return [candidate for _, candidate in decorated]


def select_exam(
    exam_candidates: Sequence[SeriesCandidate],
    anchor: SeriesCandidate,
    exam_date: str,
    config: DatasetConfig,
    min_ispy2_similarity: float,
    deep_phase_check: bool,
    candidate_phase_counts: bool,
    candidate_audit: str,
) -> SelectedExam:
    if config.dataset == "ISPY2":
        return select_ispy2_exam(
            exam_candidates=exam_candidates,
            anchor=anchor,
            exam_date=exam_date,
            config=config,
            min_similarity=min_ispy2_similarity,
            deep_phase_check=deep_phase_check,
            candidate_phase_counts=candidate_phase_counts,
            candidate_audit=candidate_audit,
        )
    return select_ispy1_or_nact_exam(
        exam_candidates=exam_candidates,
        anchor=anchor,
        exam_date=exam_date,
        config=config,
        deep_phase_check=deep_phase_check,
        candidate_phase_counts=candidate_phase_counts,
        candidate_audit=candidate_audit,
    )


def select_manual_phase_manifest(
    manifest: ManualPhaseManifest,
    exam_candidates: Sequence[SeriesCandidate],
    anchor: SeriesCandidate,
    config: DatasetConfig,
    candidate_phase_counts: bool,
    candidate_audit: str,
) -> SelectedExam:
    candidate_by_path = {
        str(candidate.series_dir.resolve()): candidate
        for candidate in exam_candidates
    }
    selected_candidates: list[SeriesCandidate] = []
    selected_paths: set[str] = set()
    phases: list[list[Path]] = []
    missing_dirs: list[str] = []
    for phase_index in sorted(manifest.phase_dirs):
        phase_files: list[Path] = []
        for series_dir in manifest.phase_dirs[phase_index]:
            series_key = str(series_dir.resolve())
            candidate = candidate_by_path.get(series_key)
            if candidate is not None and series_key not in selected_paths:
                selected_candidates.append(candidate)
                selected_paths.add(series_key)
            files = list_dicom_files_sorted(series_dir)
            if not files:
                missing_dirs.append(str(series_dir))
            phase_files.extend(files)
        phases.append(phase_files)

    rows = []
    if candidate_audit in {"all", "selected"}:
        anchor_family = normalize_sequence_family(anchor.series_description)
        for candidate in selected_candidates:
            rows.append(
                ispy1_nact_candidate_row(
                    candidate,
                    manifest.exam_date,
                    config,
                    anchor_family,
                    selected=True,
                    compute_phase_count=candidate_phase_counts,
                )
            )

    note = f"manual_phase_manifest:{manifest.path}"
    if missing_dirs:
        note += f"; missing_series_dirs={len(missing_dirs)}"
    return SelectedExam(
        exam_date=manifest.exam_date,
        exam_dir=exam_candidates[0].exam_dir if exam_candidates else None,
        selected_candidates=selected_candidates,
        selection_note=note,
        selection_score=None,
        phase_source_kind="manual_phase_manifest",
        phases=phases,
        candidate_rows=rows,
    )


def reconstructed_phase_paths(exam_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(exam_dir.glob("volume_phase*.nii.gz")):
        match = re.search(r"volume_phase(\d+)\.nii\.gz$", path.name)
        if match:
            paths[int(match.group(1))] = path
    return paths


def quarantined_patient_roots(
    reconstructed_root: Path,
    reconstructed_dir: str,
    recon_id: str,
) -> list[Path]:
    quarantine_root = reconstructed_root / "Quarantine"
    if not quarantine_root.exists():
        return []
    roots = []
    for quarantine_case in sorted(path for path in quarantine_root.iterdir() if path.is_dir()):
        candidate = quarantine_case / reconstructed_dir / recon_id
        if candidate.is_dir():
            roots.append(candidate)
    return roots


def patient_root_status(active_root: Path, quarantine_roots: Sequence[Path]) -> str:
    if active_root.exists() and quarantine_roots:
        return "active_and_quarantined"
    if active_root.exists():
        return "active"
    if quarantine_roots:
        return "quarantined_not_active"
    return "missing"


def quarantined_exam_dirs(quarantine_roots: Sequence[Path]) -> dict[str, Path]:
    exam_dirs: dict[str, Path] = {}
    for root in quarantine_roots:
        for exam_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            exam_dirs.setdefault(exam_dir.name, exam_dir)
    return exam_dirs


def mamamia_phase_paths(mamamia_root: Path, patient_id: str) -> dict[int, Path]:
    image_dir = mamamia_root / "images" / patient_id
    paths: dict[int, Path] = {}
    for path in sorted(image_dir.glob(f"{patient_id}_*.nii.gz")):
        phase_text = path.name.removeprefix(f"{patient_id}_").removesuffix(".nii.gz")
        if phase_text.isdigit():
            paths[int(phase_text)] = path
    return paths


def nifti_record(path: Path | None, prefix: str) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            f"{prefix}_path": str(path) if path is not None else "",
            f"{prefix}_exists": False,
            f"{prefix}_load_error": "",
        }
    try:
        image = nib.load(str(path))
        shape = tuple(int(value) for value in image.shape[:3])
        spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
        axcodes = "".join(nib.orientations.aff2axcodes(image.affine))
        return {
            f"{prefix}_path": str(path),
            f"{prefix}_exists": True,
            f"{prefix}_load_error": "",
            f"{prefix}_shape": "x".join(str(value) for value in shape),
            f"{prefix}_spacing": "x".join(f"{value:.6g}" for value in spacing),
            f"{prefix}_axcodes": axcodes,
            f"{prefix}_affine": json.dumps(np.asarray(image.affine).round(6).tolist()),
        }
    except Exception as exc:
        return {
            f"{prefix}_path": str(path),
            f"{prefix}_exists": True,
            f"{prefix}_load_error": str(exc),
        }


def load_nifti_header(path: Path) -> tuple[tuple[int, ...], tuple[float, ...], np.ndarray] | None:
    try:
        image = nib.load(str(path))
        return (
            tuple(int(value) for value in image.shape[:3]),
            tuple(float(value) for value in image.header.get_zooms()[:3]),
            np.asarray(image.affine, dtype=float),
        )
    except Exception:
        return None


def all_same(items: Sequence[object]) -> bool:
    return len(set(items)) <= 1


def affines_all_close(affines: Sequence[np.ndarray], atol: float) -> bool:
    if not affines:
        return True
    first = affines[0]
    return all(np.allclose(first, affine, atol=atol) for affine in affines[1:])


def phase_indices_status(indices: Sequence[int]) -> str:
    if not indices:
        return "missing"
    expected = list(range(min(indices), max(indices) + 1))
    if list(indices) != expected:
        return "non_contiguous"
    if min(indices) != 0:
        return "does_not_start_at_zero"
    return "contiguous_from_zero"


def compare_tuple_strings(left: str, right: str) -> bool:
    return bool(left and right and left == right)


def nifti_data_fingerprint(path: Path) -> str:
    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    array = np.ascontiguousarray(data)
    digest = hashlib.sha1()
    digest.update(str(array.shape).encode("utf-8"))
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def duplicate_reconstructed_phase_notes(
    phase_paths: dict[int, Path],
    exam_date: str,
    seen_fingerprints: dict[tuple[int, str], list[str]],
) -> str:
    notes = []
    for phase_index, path in sorted(phase_paths.items()):
        try:
            fingerprint = nifti_data_fingerprint(path)
        except Exception as exc:
            notes.append(f"phase{phase_index}:fingerprint_error:{exc}")
            continue
        key = (phase_index, fingerprint)
        previous_dates = seen_fingerprints.get(key, [])
        if previous_dates:
            notes.append(f"phase{phase_index}:duplicates:{';'.join(previous_dates)}")
        previous_dates.append(exam_date)
        seen_fingerprints[key] = previous_dates
    return " | ".join(notes)


def max_abs_affine_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def downsampled_array(data: np.ndarray, stride: Sequence[int]) -> np.ndarray:
    slices = tuple(slice(None, None, max(1, int(step))) for step in stride[: data.ndim])
    return np.asarray(data[slices], dtype=np.float64)


def corrcoef_flat(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return float("nan")
    return float(np.dot(x, y) / denom)


def normalized_rmse(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float64).ravel()
    y = np.asarray(right, dtype=np.float64).ravel()
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    x = (x - float(np.mean(x))) / x_std
    y = (y - float(np.mean(y))) / y_std
    return float(np.sqrt(np.mean((x - y) ** 2)))


def transform_candidates(array: np.ndarray, target_shape: tuple[int, ...]) -> Iterable[tuple[str, np.ndarray]]:
    base_transforms = [("identity", array)]
    if array.ndim >= 2:
        base_transforms.append(("swap_xy", np.swapaxes(array, 0, 1)))
    for base_name, base in base_transforms:
        axis_count = min(3, base.ndim)
        for count in range(axis_count + 1):
            for axes in combinations(range(axis_count), count):
                if not axes:
                    name = base_name
                    transformed = base
                else:
                    suffix = "flip_" + "".join(str(axis) for axis in axes)
                    name = f"{base_name}+{suffix}" if base_name != "identity" else suffix
                    transformed = np.flip(base, axis=axes)
                if transformed.shape[: len(target_shape)] == target_shape:
                    yield name, transformed


def image_similarity(
    reconstructed_path: Path,
    mamamia_path: Path,
    stride: Sequence[int],
) -> dict[str, object]:
    recon_img = nib.load(str(reconstructed_path))
    mamamia_img = nib.load(str(mamamia_path))
    recon = np.asanyarray(recon_img.dataobj)
    mamamia = np.asanyarray(mamamia_img.dataobj)
    if recon.shape[:3] != mamamia.shape[:3]:
        return {
            "pixel_compare_status": "shape_mismatch",
            "identity_corr": "",
            "best_corr": "",
            "best_transform": "",
            "best_nrmse": "",
        }
    target_shape = mamamia.shape[:3]
    mamamia_sample = downsampled_array(mamamia, stride)
    best_corr = -math.inf
    best_nrmse = float("nan")
    best_transform = ""
    identity_corr = ""
    for name, transformed in transform_candidates(recon, target_shape):
        sample = downsampled_array(transformed, stride)
        corr = corrcoef_flat(sample, mamamia_sample)
        nrmse = normalized_rmse(sample, mamamia_sample)
        if name == "identity":
            identity_corr = corr
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_nrmse = nrmse
            best_transform = name
    if best_corr >= 0.999:
        status = f"pixel_equivalent_after_{best_transform}"
    elif best_corr >= 0.98:
        status = f"very_similar_after_{best_transform}"
    elif best_corr >= 0.90:
        status = f"similar_after_{best_transform}"
    else:
        status = "low_similarity"
    return {
        "pixel_compare_status": status,
        "identity_corr": round(float(identity_corr), 6) if identity_corr != "" else "",
        "best_corr": round(float(best_corr), 6) if np.isfinite(best_corr) else "",
        "best_transform": best_transform,
        "best_nrmse": round(float(best_nrmse), 6) if np.isfinite(best_nrmse) else "",
    }


def phase_header_records(
    phase_paths: dict[int, Path],
    affine_atol: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    shapes: list[str] = []
    spacings: list[str] = []
    affines: list[np.ndarray] = []
    load_errors: list[str] = []
    for phase_index, path in sorted(phase_paths.items()):
        record = {"phase_index": phase_index}
        record.update(nifti_record(path, "reconstructed"))
        rows.append(record)
        if record.get("reconstructed_load_error"):
            load_errors.append(str(record["reconstructed_load_error"]))
            continue
        shapes.append(str(record.get("reconstructed_shape", "")))
        spacings.append(str(record.get("reconstructed_spacing", "")))
        affine_text = record.get("reconstructed_affine", "")
        if affine_text:
            affines.append(np.asarray(json.loads(str(affine_text)), dtype=float))
    summary = {
        "reconstructed_num_phases": len(phase_paths),
        "reconstructed_phase_indices": ";".join(str(index) for index in sorted(phase_paths)),
        "reconstructed_phase_indices_status": phase_indices_status(sorted(phase_paths)),
        "reconstructed_phase_shapes": ";".join(sorted(set(shapes))),
        "reconstructed_phase_spacings": ";".join(sorted(set(spacings))),
        "reconstructed_shapes_consistent": all_same(shapes),
        "reconstructed_spacings_consistent": all_same(spacings),
        "reconstructed_affines_consistent": affines_all_close(affines, affine_atol),
        "reconstructed_header_errors": ";".join(load_errors),
    }
    return rows, summary


def compare_anchor_phase(
    patient_id: str,
    dataset: str,
    phase_index: int,
    reconstructed_path: Path | None,
    mamamia_path: Path | None,
    compare_pixels: bool,
    stride: Sequence[int],
    affine_atol: float,
    spacing_atol: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "patient_id": patient_id,
        "dataset": dataset,
        "phase_index": phase_index,
    }
    row.update(nifti_record(reconstructed_path, "reconstructed"))
    row.update(nifti_record(mamamia_path, "mamamia"))
    recon_header = load_nifti_header(reconstructed_path) if reconstructed_path else None
    mamamia_header = load_nifti_header(mamamia_path) if mamamia_path else None
    if reconstructed_path is None or not reconstructed_path.exists():
        row["anchor_compare_status"] = "missing_reconstructed_phase"
        return row
    if mamamia_path is None or not mamamia_path.exists():
        row["anchor_compare_status"] = "missing_mamamia_phase"
        return row
    if recon_header is None or mamamia_header is None:
        row["anchor_compare_status"] = "header_load_error"
        return row
    recon_shape, recon_spacing, recon_affine = recon_header
    mamamia_shape, mamamia_spacing, mamamia_affine = mamamia_header
    row["shape_matches"] = recon_shape == mamamia_shape
    row["spacing_matches"] = np.allclose(recon_spacing, mamamia_spacing, atol=spacing_atol)
    row["affine_matches"] = np.allclose(recon_affine, mamamia_affine, atol=affine_atol)
    row["max_abs_affine_diff"] = round(max_abs_affine_diff(recon_affine, mamamia_affine), 6)
    if compare_pixels:
        try:
            row.update(image_similarity(reconstructed_path, mamamia_path, stride))
        except Exception as exc:
            row["pixel_compare_status"] = f"pixel_compare_error:{exc}"
    if not row["shape_matches"]:
        row["anchor_compare_status"] = "shape_mismatch"
    elif not row["spacing_matches"]:
        row["anchor_compare_status"] = "spacing_mismatch"
    elif row.get("pixel_compare_status", "").startswith("pixel_equivalent"):
        row["anchor_compare_status"] = "same_content_after_orientation_transform"
    elif row["affine_matches"]:
        row["anchor_compare_status"] = "same_grid"
    elif compare_pixels and row.get("pixel_compare_status") == "low_similarity":
        row["anchor_compare_status"] = "same_shape_low_similarity"
    else:
        row["anchor_compare_status"] = "same_shape_spacing_affine_diff"
    return row


def selected_series_summary(selected: SelectedExam) -> dict[str, object]:
    candidates = selected.selected_candidates
    descriptions = [candidate.series_description for candidate in candidates]
    dirs = [str(candidate.series_dir) for candidate in candidates]
    uids = [candidate.series_uid for candidate in candidates]
    num_images = [
        "" if candidate.num_images is None else str(candidate.num_images)
        for candidate in candidates
    ]
    planes = [
        get_orientation_label(first_dicom_header(candidate.series_dir), candidate.series_description)
        for candidate in candidates
    ]
    return {
        "selected_num_series": len(candidates),
        "selected_series_uids": " | ".join(uids),
        "selected_series_descriptions": " | ".join(descriptions),
        "selected_series_dirs": " | ".join(dirs),
        "selected_num_images_metadata": " | ".join(num_images),
        "selected_planes": " | ".join(planes),
    }


def expected_phase_summary(selected: SelectedExam) -> dict[str, object]:
    if not selected.phases:
        return {
            "expected_num_phases_from_dicom": "",
            "expected_phase_slice_counts": "",
            "expected_phase_shapes": "",
            "expected_phase_spacings": "",
        }
    shapes: list[str] = []
    spacings: list[str] = []
    counts: list[str] = []
    for phase_files in selected.phases:
        shape, spacing = dicom_spacing_and_shape(phase_files)
        counts.append(str(len(phase_files)))
        shapes.append("x".join(str(value) for value in shape))
        spacings.append("x".join(f"{value:.6g}" for value in spacing))
    return {
        "expected_num_phases_from_dicom": len(selected.phases),
        "expected_phase_slice_counts": ";".join(counts),
        "expected_phase_shapes": ";".join(shapes),
        "expected_phase_spacings": ";".join(spacings),
        "expected_phase_shapes_unique": ";".join(sorted(set(shapes))),
        "expected_phase_spacings_unique": ";".join(sorted(set(spacings))),
    }


def compare_expected_to_reconstructed(
    selected: SelectedExam,
    reconstructed_paths: dict[int, Path],
    spacing_atol: float,
) -> dict[str, object]:
    if not selected.phases:
        return {
            "phase_count_matches_expected": "",
            "phase_header_matches_expected": "",
            "expected_missing_phase_indices": "",
            "unexpected_reconstructed_phase_indices": "",
        }
    expected_indices = set(range(len(selected.phases)))
    reconstructed_indices = set(reconstructed_paths)
    missing = sorted(expected_indices - reconstructed_indices)
    unexpected = sorted(reconstructed_indices - expected_indices)
    header_matches = True
    header_notes: list[str] = []
    for phase_index, phase_files in enumerate(selected.phases):
        recon_path = reconstructed_paths.get(phase_index)
        if recon_path is None:
            header_matches = False
            continue
        expected_shape, expected_spacing = dicom_spacing_and_shape(phase_files)
        recon_header = load_nifti_header(recon_path)
        if recon_header is None:
            header_matches = False
            header_notes.append(f"phase{phase_index}:recon_header_error")
            continue
        recon_shape, recon_spacing, _ = recon_header
        if expected_shape and tuple(recon_shape) != tuple(expected_shape):
            header_matches = False
            header_notes.append(
                f"phase{phase_index}:shape {recon_shape} != expected {expected_shape}"
            )
        if expected_spacing and not np.allclose(recon_spacing, expected_spacing, atol=spacing_atol):
            header_matches = False
            header_notes.append(
                f"phase{phase_index}:spacing {recon_spacing} != expected {expected_spacing}"
            )
    return {
        "phase_count_matches_expected": len(reconstructed_paths) == len(selected.phases),
        "phase_header_matches_expected": header_matches,
        "expected_missing_phase_indices": ";".join(str(index) for index in missing),
        "unexpected_reconstructed_phase_indices": ";".join(str(index) for index in unexpected),
        "expected_vs_reconstructed_notes": " | ".join(header_notes),
    }


def risk_status(
    *,
    is_anchor: bool,
    mamamia_phase_count: int,
    selected: SelectedExam,
    phase_summary: dict[str, object],
    expected_comparison: dict[str, object],
    mamamia_view: str,
    reconstructed_exam_exists: bool,
    quarantined_exam_exists: bool,
    duplicate_phase_notes: str,
) -> str:
    issues: list[str] = []
    if not selected.selected_candidates:
        issues.append("no_selected_source_series")
    if phase_summary["reconstructed_phase_indices_status"] != "contiguous_from_zero":
        if not reconstructed_exam_exists and quarantined_exam_exists:
            issues.append("reconstruction_quarantined_not_active")
        else:
            issues.append(str(phase_summary["reconstructed_phase_indices_status"]))
    if not bool(phase_summary["reconstructed_shapes_consistent"]):
        issues.append("shape_mismatch_between_reconstructed_phases")
    if not bool(phase_summary["reconstructed_spacings_consistent"]):
        issues.append("spacing_mismatch_between_reconstructed_phases")
    if not bool(phase_summary["reconstructed_affines_consistent"]):
        issues.append("affine_mismatch_between_reconstructed_phases")
    if selected.phases:
        if expected_comparison.get("phase_count_matches_expected") is False:
            issues.append("phase_count_mismatch_vs_selected_dicom")
        if expected_comparison.get("phase_header_matches_expected") is False:
            issues.append("phase_header_mismatch_vs_selected_dicom")
    if is_anchor and mamamia_phase_count and reconstructed_exam_exists:
        if phase_summary["reconstructed_num_phases"] != mamamia_phase_count:
            issues.append("anchor_phase_count_mismatch_vs_mamamia")
    selected_planes = {
        plane.strip()
        for plane in str(selected_series_summary(selected).get("selected_planes", "")).split("|")
        if plane.strip()
    }
    if mamamia_view and selected_planes and mamamia_view not in selected_planes:
        issues.append(f"view_mismatch_vs_mamamia:{mamamia_view}")
    if selected.selection_note.startswith("no_family_match") or "second_best_score" in selected.selection_note:
        issues.append("low_or_ambiguous_series_selection")
    if duplicate_phase_notes:
        if "duplicates:" in duplicate_phase_notes:
            issues.append("duplicate_reconstructed_volume")
        if "fingerprint_error:" in duplicate_phase_notes:
            issues.append("duplicate_check_error")
    return "ok" if not issues else "|".join(sorted(set(issues)))


def build_patient_rows(args: argparse.Namespace) -> pd.DataFrame:
    clinical_path = args.mamamia_root / "clinical_and_imaging_info.xlsx"
    clinical = pd.read_excel(clinical_path, sheet_name="dataset_info")
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    clinical["dataset"] = clinical["dataset"].astype(str)
    clinical["anchor_reconstructed_date"] = clinical["acquisition_date"].map(date_to_folder)
    clinical = clinical[clinical["dataset"].isin(set(args.datasets))].copy()
    if args.patients:
        wanted = {str(patient) for patient in args.patients}
        clinical = clinical[clinical["patient_id"].isin(wanted)].copy()
    if args.max_patients_per_dataset is not None:
        clinical = (
            clinical.sort_values(["dataset", "patient_id"], kind="stable")
            .groupby("dataset", group_keys=False)
            .head(args.max_patients_per_dataset)
            .copy()
        )
    return clinical.sort_values(["dataset", "patient_id"], kind="stable").reset_index(drop=True)


def should_include_exam(timepoints: str, is_anchor: bool) -> bool:
    if timepoints == "all":
        return True
    if timepoints == "anchor":
        return is_anchor
    return not is_anchor


def audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    patient_rows = build_patient_rows(args)
    metadata_cache: dict[str, tuple[dict[str, list[SeriesCandidate]], dict[str, SeriesCandidate]]] = {}
    manual_manifests = load_manual_phase_manifests(args.manual_phase_manifest_dir)
    exam_rows: list[dict[str, object]] = []
    candidate_rows: list[dict[str, object]] = []
    anchor_rows: list[dict[str, object]] = []
    anchor_pixel_count = 0

    for dataset in args.datasets:
        config = DATASET_CONFIGS[dataset]
        metadata_cache[dataset] = load_metadata(config)

    total_patients = len(patient_rows)
    for patient_number, (_, patient_row) in enumerate(patient_rows.iterrows(), start=1):
        dataset = str(patient_row["dataset"])
        config = DATASET_CONFIGS[dataset]
        by_subject, by_uid = metadata_cache[dataset]
        patient_id = str(patient_row["patient_id"])
        if args.progress_every and (
            patient_number == 1 or patient_number % args.progress_every == 0
        ):
            print(
                f"[audit] patient {patient_number}/{total_patients}: {patient_id} ({dataset})",
                flush=True,
            )
        metadata_id = metadata_subject_id(patient_id, dataset)
        recon_id = reconstructed_patient_id(patient_id, dataset)
        anchor_uid = clean_uid(patient_row.get("tcia_series_uid", ""))
        anchor_date = str(patient_row.get("anchor_reconstructed_date", ""))
        mamamia_view = str(patient_row.get("view", "") or "").strip().lower()
        mamamia_phases = mamamia_phase_paths(args.mamamia_root, patient_id)
        mamamia_phase_count = len(mamamia_phases)

        anchor_candidate = by_uid.get(anchor_uid)
        patient_candidates = by_subject.get(metadata_id, [])
        exams_by_date = group_by_exam(patient_candidates)
        recon_root = args.reconstructed_root / config.reconstructed_dir / recon_id
        quarantine_roots = quarantined_patient_roots(
            args.reconstructed_root,
            config.reconstructed_dir,
            recon_id,
        )
        quarantine_exam_dirs = quarantined_exam_dirs(quarantine_roots)
        recon_root_status = patient_root_status(recon_root, quarantine_roots)

        if anchor_candidate is None or not patient_candidates:
            exam_rows.append(
                {
                    "patient_id": patient_id,
                    "dataset": dataset,
                    "metadata_patient_id": metadata_id,
                    "reconstructed_patient_id": recon_id,
                    "anchor_uid": anchor_uid,
                    "anchor_reconstructed_date": anchor_date,
                    "exam_date": "",
                    "is_anchor_date": "",
                    "risk_status": "missing_anchor_uid_or_metadata_patient",
                    "reconstructed_patient_root": str(recon_root),
                    "reconstructed_patient_root_exists": recon_root.exists(),
                    "reconstructed_patient_root_status": recon_root_status,
                    "quarantined_patient_roots": " | ".join(
                        str(path) for path in quarantine_roots
                    ),
                }
            )
            continue

        reconstructed_exam_dirs = {
            path.name: path for path in sorted(recon_root.iterdir()) if path.is_dir()
        } if recon_root.exists() else {}
        if args.include_metadata_only_exams:
            all_exam_dates = sorted(set(exams_by_date) | set(reconstructed_exam_dirs))
        else:
            required_dates = set(reconstructed_exam_dirs)
            if anchor_date:
                required_dates.add(anchor_date)
            all_exam_dates = sorted(required_dates)

        seen_phase_fingerprints: dict[tuple[int, str], list[str]] = {}
        for exam_date in all_exam_dates:
            is_anchor = exam_date == anchor_date
            if not should_include_exam(args.timepoints, is_anchor):
                continue
            exam_candidates = exams_by_date.get(exam_date, [])
            recon_exam_dir = reconstructed_exam_dirs.get(exam_date)
            quarantine_exam_dir = quarantine_exam_dirs.get(exam_date)
            phase_paths = reconstructed_phase_paths(recon_exam_dir) if recon_exam_dir else {}
            _, phase_summary = phase_header_records(phase_paths, args.affine_atol)
            duplicate_notes = ""
            if args.check_duplicate_reconstructed_volumes:
                duplicate_notes = duplicate_reconstructed_phase_notes(
                    phase_paths,
                    exam_date,
                    seen_phase_fingerprints,
                )

            if not exam_candidates:
                selected = SelectedExam(
                    exam_date=exam_date,
                    exam_dir=None,
                    selected_candidates=[],
                    selection_note="no_metadata_exam_for_reconstructed_date",
                    selection_score=None,
                    phase_source_kind="",
                    phases=[],
                    candidate_rows=[],
                )
            elif (patient_id, exam_date) in manual_manifests:
                selected = select_manual_phase_manifest(
                    manual_manifests[(patient_id, exam_date)],
                    exam_candidates=exam_candidates,
                    anchor=anchor_candidate,
                    config=config,
                    candidate_phase_counts=args.candidate_phase_counts,
                    candidate_audit=args.candidate_audit,
                )
            else:
                selected = select_exam(
                    exam_candidates=exam_candidates,
                    anchor=anchor_candidate,
                    exam_date=exam_date,
                    config=config,
                    min_ispy2_similarity=args.min_ispy2_family_similarity,
                    deep_phase_check=args.deep_dicom_phase_check,
                    candidate_phase_counts=args.candidate_phase_counts,
                    candidate_audit=args.candidate_audit,
                )

            selected_summary = selected_series_summary(selected)
            expected_summary = expected_phase_summary(selected)
            expected_comparison = compare_expected_to_reconstructed(
                selected,
                phase_paths,
                spacing_atol=args.spacing_atol,
            )
            current_risk = risk_status(
                is_anchor=is_anchor,
                mamamia_phase_count=mamamia_phase_count,
                selected=selected,
                phase_summary=phase_summary,
                expected_comparison=expected_comparison,
                mamamia_view=mamamia_view,
                reconstructed_exam_exists=recon_exam_dir is not None,
                quarantined_exam_exists=quarantine_exam_dir is not None,
                duplicate_phase_notes=duplicate_notes,
            )

            base_row: dict[str, object] = {
                "patient_id": patient_id,
                "dataset": dataset,
                "metadata_patient_id": metadata_id,
                "reconstructed_patient_id": recon_id,
                "anchor_uid": anchor_uid,
                "anchor_series_description": anchor_candidate.series_description,
                "anchor_series_dir": str(anchor_candidate.series_dir),
                "anchor_reconstructed_date": anchor_date,
                "exam_date": exam_date,
                "is_anchor_date": is_anchor,
                "mamamia_num_phases": mamamia_phase_count,
                "mamamia_view": mamamia_view,
                "metadata_exam_exists": bool(exam_candidates),
                "reconstructed_exam_exists": recon_exam_dir is not None,
                "reconstructed_exam_dir": str(recon_exam_dir) if recon_exam_dir else "",
                "reconstructed_patient_root_status": recon_root_status,
                "quarantined_patient_roots": " | ".join(
                    str(path) for path in quarantine_roots
                ),
                "quarantined_exam_exists": quarantine_exam_dir is not None,
                "quarantined_exam_dir": str(quarantine_exam_dir) if quarantine_exam_dir else "",
                "selection_note": selected.selection_note,
                "selection_score": "" if selected.selection_score is None else round(selected.selection_score, 6),
                "phase_source_kind": selected.phase_source_kind,
                "risk_status": current_risk,
                "duplicate_reconstructed_phase_notes": duplicate_notes,
                "manual_phase_manifest": (
                    str(manual_manifests[(patient_id, exam_date)].path)
                    if (patient_id, exam_date) in manual_manifests
                    else ""
                ),
            }
            base_row.update(selected_summary)
            base_row.update(expected_summary)
            base_row.update(phase_summary)
            base_row.update(expected_comparison)
            exam_rows.append(base_row)

            for row in selected.candidate_rows:
                row = {
                    "patient_id": patient_id,
                    "metadata_patient_id": metadata_id,
                    "reconstructed_patient_id": recon_id,
                    "anchor_uid": anchor_uid,
                    "anchor_series_description": anchor_candidate.series_description,
                    **row,
                }
                candidate_rows.append(row)

            if is_anchor:
                for phase_index in sorted(set(phase_paths) | set(mamamia_phases)):
                    compare_pixels = bool(args.compare_anchor_pixels)
                    if args.max_anchor_pixel_comparisons is not None:
                        compare_pixels = (
                            compare_pixels
                            and anchor_pixel_count < args.max_anchor_pixel_comparisons
                        )
                    anchor_rows.append(
                        compare_anchor_phase(
                            patient_id=patient_id,
                            dataset=dataset,
                            phase_index=phase_index,
                            reconstructed_path=phase_paths.get(phase_index),
                            mamamia_path=mamamia_phases.get(phase_index),
                            compare_pixels=compare_pixels,
                            stride=args.similarity_stride,
                            affine_atol=args.affine_atol,
                            spacing_atol=args.spacing_atol,
                        )
                    )
                    if compare_pixels:
                        anchor_pixel_count += 1

    exam_df = pd.DataFrame(exam_rows)
    candidate_df = pd.DataFrame(candidate_rows)
    anchor_df = pd.DataFrame(anchor_rows)
    summary = {
        "mamamia_root": str(args.mamamia_root.resolve()),
        "reconstructed_root": str(args.reconstructed_root.resolve()),
        "datasets": list(args.datasets),
        "patients_audited": int(len(patient_rows)),
        "timepoints": args.timepoints,
        "deep_dicom_phase_check": bool(args.deep_dicom_phase_check),
        "candidate_phase_counts": bool(args.candidate_phase_counts),
        "candidate_audit": args.candidate_audit,
        "compare_anchor_pixels": bool(args.compare_anchor_pixels),
        "include_metadata_only_exams": bool(args.include_metadata_only_exams),
        "manual_phase_manifest_dir": str(args.manual_phase_manifest_dir.resolve()),
        "manual_phase_manifests_loaded": int(len(manual_manifests)),
        "exam_rows": int(len(exam_df)),
        "anchor_phase_rows": int(len(anchor_df)),
        "candidate_rows": int(len(candidate_df)),
        "risk_status_counts": compact_counts(exam_df.get("risk_status", [])),
        "anchor_compare_status_counts": compact_counts(
            anchor_df.get("anchor_compare_status", [])
        ),
        "anchor_pixel_compare_status_counts": compact_counts(
            anchor_df.get("pixel_compare_status", [])
        ),
        "notes": [
            "ok means the selected source metadata, reconstructed phase count, phase headers, and view checks passed for the checks enabled.",
            "Anchor pixel equivalence is checked only when --compare-anchor-pixels is used.",
            "Extra-timepoint correctness is based on re-selecting source series from metadata/DICOM headers using the reconstruction rules and flagging low-confidence or ambiguous choices.",
            "Shape differences versus MAMA-MIA may reflect MAMA-MIA preprocessing, but they must be documented before using reconstructed anchors interchangeably.",
            "Use --include-metadata-only-exams when you also want to find original exams that were not reconstructed.",
        ],
    }
    return exam_df, candidate_df, anchor_df, summary


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    exam_df, candidate_df, anchor_df, summary = audit(args)

    exam_path = output_root / "reconstructed_exam_audit.csv"
    candidate_path = output_root / "series_candidate_audit.csv"
    anchor_path = output_root / "anchor_phase_image_audit.csv"
    summary_path = output_root / "summary.json"

    exam_df.to_csv(exam_path, index=False)
    candidate_df.to_csv(candidate_path, index=False)
    anchor_df.to_csv(anchor_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Patients audited: {summary['patients_audited']}")
    print(f"Exam rows:        {summary['exam_rows']}")
    print(f"Anchor rows:      {summary['anchor_phase_rows']}")
    print(f"Candidate rows:   {summary['candidate_rows']}")
    print("Risk status counts:")
    for key, value in summary["risk_status_counts"].items():
        print(f"  {key}: {value}")
    print("Anchor comparison counts:")
    for key, value in summary["anchor_compare_status_counts"].items():
        print(f"  {key}: {value}")
    print(f"Exam audit:       {exam_path}")
    print(f"Candidate audit:  {candidate_path}")
    print(f"Anchor audit:     {anchor_path}")
    print(f"Summary:          {summary_path}")


if __name__ == "__main__":
    main()
