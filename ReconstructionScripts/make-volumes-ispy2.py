import argparse
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk


DEFAULT_DATASET_ROOT = Path("data")
DEFAULT_EXCEL_PATH = DEFAULT_DATASET_ROOT / "clinical_and_imaging_info.xlsx"
DEFAULT_CSV_PATH = DEFAULT_DATASET_ROOT / "ISPY-2/manifest-1641168072464/metadata.csv"
DEFAULT_OUTPUT_ROOT = Path("Reconstructed_Datasets/ISPY2-Volumes")
DEFAULT_REPORT_CSV = DEFAULT_DATASET_ROOT / "DatasetVolumeCreation/ispy2_volume_construction_report.csv"

SERIES_PHASE_STOPWORDS = {
    "aligned",
    "contrast",
    "inject",
    "injection",
    "registered",
    "repeat",
    "test",
    "to",
    "with",
}

PHASE_ONLY_DYNAMIC_TOKEN = "phase_only_dynamic"


@dataclass
class SeriesCandidate:
    patient_id: str
    exam_key: str
    exam_dir: Path
    study_date: str
    study_description: str
    series_uid: str
    series_dir: Path
    series_description: str
    modality: str
    family_key: str
    family_tokens: Set[str]
    num_images: Optional[int]


@dataclass
class ExamSelection:
    exam_key: str
    exam_dir: Path
    exam_date: str
    study_description: str
    candidates: List[SeriesCandidate]
    note: str
    selected_family_key: str
    selected_score: float


def clean_uid(value) -> str:
    return str(value).strip()


def normalize_ispy2_patient_id(value) -> str:
    return str(value).strip().upper().replace("_", "-")


def extract_exam_date_from_exam_folder(exam_folder_name: str) -> str:
    match = re.match(r"^(\d{2}-\d{2}-\d{4})", exam_folder_name)
    if match:
        return match.group(1)
    return "unknown_date"


def list_dicom_files_sorted(dicom_dir: Path) -> List[Path]:
    files = [path for path in dicom_dir.iterdir() if path.is_file()]
    files.sort(key=lambda path: path.name)
    return files


def parse_time_to_number(value) -> Optional[int]:
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"[^0-9.]", "", text)
    if not text:
        return None

    try:
        return int(float(text))
    except ValueError:
        return None


def is_phase_only_dynamic_series(text: str) -> bool:
    normalized = text.strip()
    normalized = re.sub(r"ispy\s*2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"ispy2", " ", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized.lower()).strip()
    if not normalized:
        return False

    tokens = normalized.split()
    if not tokens:
        return False

    return all(re.fullmatch(r"(pre|post\d*|ph\d+|phase\d+)", token) for token in tokens)


def get_phase_time_number(ds) -> Optional[int]:
    acquisition_time = parse_time_to_number(getattr(ds, "AcquisitionTime", None))
    if acquisition_time is not None:
        return acquisition_time

    content_time = parse_time_to_number(getattr(ds, "ContentTime", None))
    if content_time is not None:
        return content_time

    return None


def get_phase_min_time(phase_files: Sequence[Path]) -> Optional[int]:
    values = []

    for path in phase_files:
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            continue

        value = get_phase_time_number(ds)
        if value is not None:
            values.append(value)

    if not values:
        return None

    return min(values)


def split_into_phases_by_acquisition_time(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> List[List[Path]]:
    if not dicom_files:
        return []

    phases: List[List[Path]] = []
    current_phase: List[Path] = []
    previous_time: Optional[int] = None

    for path in dicom_files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        current_time = get_phase_time_number(ds)

        if not current_phase:
            current_phase.append(path)
            previous_time = current_time
            continue

        start_new_phase = False
        if previous_time is not None and current_time is not None:
            if abs(current_time - previous_time) >= phase_gap:
                start_new_phase = True

        if start_new_phase:
            phases.append(current_phase)
            current_phase = [path]
        else:
            current_phase.append(path)

        previous_time = current_time

    if current_phase:
        phases.append(current_phase)

    decorated = []
    for index, phase_files in enumerate(phases):
        min_time = get_phase_min_time(phase_files)
        sort_key = (min_time is None, min_time if min_time is not None else index)
        decorated.append((sort_key, phase_files))

    decorated.sort(key=lambda item: item[0])
    return [phase_files for _, phase_files in decorated]


def split_into_phases_by_grouped_time(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> List[List[Path]]:
    if not dicom_files:
        return []

    indexed_times = []
    for index, path in enumerate(dicom_files):
        try:
            ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        except Exception:
            return []

        current_time = get_phase_time_number(ds)
        if current_time is None:
            return []

        indexed_times.append((index, path, current_time))

    unique_times = sorted({time_value for _, _, time_value in indexed_times})
    if not unique_times:
        return []

    time_to_cluster: Dict[int, int] = {}
    cluster_to_time: Dict[int, int] = {}
    cluster_index = -1
    previous_time = None

    for time_value in unique_times:
        if previous_time is None or abs(time_value - previous_time) >= phase_gap:
            cluster_index += 1
            cluster_to_time[cluster_index] = time_value
        time_to_cluster[time_value] = cluster_index
        previous_time = time_value

    grouped: Dict[int, List[Tuple[int, Path]]] = defaultdict(list)
    for index, path, time_value in indexed_times:
        grouped[time_to_cluster[time_value]].append((index, path))

    decorated = []
    for cluster_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        phase_files = [path for _, path in items]
        decorated.append((cluster_to_time.get(cluster_id, cluster_id), phase_files))

    decorated.sort(key=lambda item: item[0])
    return [phase_files for _, phase_files in decorated]


def split_into_phases_by_position_reset(
    dicom_files: Sequence[Path],
    tolerance: float = 1e-3,
) -> List[List[Path]]:
    if not dicom_files:
        return []

    datasets = []
    projections = []
    slice_cos = None

    for path in dicom_files:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
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
        datasets.append(ds)

    deltas = [curr - prev for prev, curr in zip(projections[:-1], projections[1:]) if abs(curr - prev) > tolerance]
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
    phases = [list(dicom_files[start:end]) for start, end in zip(boundaries[:-1], boundaries[1:])]
    phase_sizes = [len(phase_files) for phase_files in phases]
    if not phase_sizes or min(phase_sizes) < 2:
        return []

    if max(phase_sizes) - min(phase_sizes) > 2:
        return []

    return phases


def phase_size_balance(phases: Sequence[Sequence[Path]]) -> float:
    if not phases:
        return 0.0

    sizes = [len(phase_files) for phase_files in phases]
    max_size = max(sizes)
    if max_size <= 0:
        return 0.0

    return min(sizes) / max_size


def is_plausible_phase_split(phases: Sequence[Sequence[Path]]) -> bool:
    if not phases:
        return False

    sizes = [len(phase_files) for phase_files in phases]
    if min(sizes) < 2:
        return False

    if len(phases) == 1:
        return True

    return phase_size_balance(phases) >= 0.5


def split_single_series_into_phases(
    dicom_files: Sequence[Path],
    phase_gap: int,
) -> Tuple[List[List[Path]], str]:
    candidate_splits = [
        ([list(dicom_files)] if dicom_files else [], "single_series_as_is", 0),
        (split_into_phases_by_grouped_time(dicom_files, phase_gap=phase_gap), "single_series_split_by_grouped_time", 3),
        (split_into_phases_by_position_reset(dicom_files), "single_series_split_by_position_reset", 2),
        (split_into_phases_by_acquisition_time(dicom_files, phase_gap=phase_gap), "single_series_split_by_consecutive_time", 1),
    ]

    plausible_splits = [
        (phases, source_kind, priority)
        for phases, source_kind, priority in candidate_splits
        if is_plausible_phase_split(phases)
    ]

    usable_splits = plausible_splits if plausible_splits else candidate_splits
    best_phases, best_source_kind, _ = max(
        usable_splits,
        key=lambda item: (
            len(item[0]),
            phase_size_balance(item[0]),
            item[2],
        ),
    )
    return best_phases, best_source_kind


def extract_series_description_from_folder_name(series_folder_name: str) -> str:
    parts = series_folder_name.split("-")
    if len(parts) < 3:
        return series_folder_name
    return "-".join(parts[1:-1]).strip()


def normalize_series_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = text.replace(":", " ")
    text = re.sub(r"ispy\s*2", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"ispy2", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"ph\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"phase\s*\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"pre\s*--\s*registered", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"pre[-\s]*contrast", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"post[-\s]*contrast", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(prepost|delayed|delay|post|pre|registered|repeat|test)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*ml", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-z0-9]+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_series_family(text: str) -> Set[str]:
    normalized = normalize_series_text(text)
    tokens = []

    for token in normalized.split():
        if not token:
            continue
        if token in SERIES_PHASE_STOPWORDS:
            continue
        if token.isdigit():
            continue
        tokens.append(token)

    if not tokens and is_phase_only_dynamic_series(text):
        return {PHASE_ONLY_DYNAMIC_TOKEN}

    return set(tokens)


def series_family_key(text: str) -> str:
    tokens = sorted(tokenize_series_family(text))
    return " ".join(tokens)


def family_similarity(anchor_tokens: Set[str], candidate_tokens: Set[str]) -> float:
    if not anchor_tokens or not candidate_tokens:
        return 0.0

    overlap = len(anchor_tokens & candidate_tokens)
    if overlap == 0:
        return 0.0

    dice = (2.0 * overlap) / (len(anchor_tokens) + len(candidate_tokens))
    containment = overlap / len(anchor_tokens)
    return max(dice, containment)


def read_first_dataset(series_dir: Path):
    files = list_dicom_files_sorted(series_dir)
    if not files:
        return None

    try:
        return pydicom.dcmread(str(files[0]), stop_before_pixels=True, force=True)
    except Exception:
        return None


def series_min_time(series_dir: Path) -> Optional[int]:
    files = list_dicom_files_sorted(series_dir)
    if not files:
        return None
    return get_phase_min_time(files)


def series_phase_number(series_description: str) -> Optional[int]:
    match = re.search(r"ph\s*(\d+)", series_description, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    match = re.search(r"post\s*(\d+)", series_description, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))

    normalized = series_description.lower()
    if " pre " in f" {normalized} " or normalized.endswith(" pre") or "pre -- registered" in normalized:
        return 0

    return None


def sort_candidates_by_phase(candidates: Sequence[SeriesCandidate]) -> List[SeriesCandidate]:
    decorated = []

    for candidate in candidates:
        min_time = series_min_time(candidate.series_dir)
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


def build_sitk_image_from_phase_files(phase_files: Sequence[Path]) -> sitk.Image:
    datasets = [pydicom.dcmread(str(path), force=True) for path in phase_files]
    if not datasets:
        raise ValueError("Cannot build a volume from an empty phase")

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

    if len(positions) == len(datasets) and all(position is not None for position in positions):
        projected = [float(np.dot(position, slice_cos)) for position in positions]
        order = sorted(range(len(datasets)), key=lambda index: projected[index])
        datasets = [datasets[index] for index in order]
        positions = [positions[index] for index in order]
    else:
        instance_numbers = []
        for index, ds in enumerate(datasets):
            instance_number = getattr(ds, "InstanceNumber", None)
            try:
                parsed = int(instance_number)
            except Exception:
                parsed = index
            instance_numbers.append((parsed, index))
        order = [index for _, index in sorted(instance_numbers)]
        datasets = [datasets[index] for index in order]
        positions = [positions[index] for index in order]

    slices = [ds.pixel_array for ds in datasets]
    volume = np.stack(slices, axis=0)
    image = sitk.GetImageFromArray(volume)

    origin = tuple(float(x) for x in positions[0]) if positions[0] is not None else (0.0, 0.0, 0.0)

    spacing_z = None
    if len(positions) >= 2 and all(position is not None for position in positions):
        projected = [float(np.dot(position, slice_cos)) for position in positions]
        deltas = np.diff(projected)
        nonzero_deltas = [abs(delta) for delta in deltas if abs(delta) > 1e-8]
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

    image.SetSpacing((spacing_x, spacing_y, spacing_z))
    image.SetOrigin(origin)
    image.SetDirection(direction)
    return image


def add_report_row(
    report_rows: List[Dict[str, object]],
    *,
    workbook_patient_id: str,
    patient_id: str,
    anchor_uid: str,
    anchor_series_description: str,
    exam_date: str,
    study_description: str,
    exam_dir: Optional[Path],
    selected_series_descriptions: Sequence[str],
    selected_series_dirs: Sequence[Path],
    selection_note: str,
    selection_score: object,
    phase_source_kind: str,
    phase_index: object,
    num_selected_series: object,
    num_dicom_files: object,
    num_phases: object,
    num_slices_in_phase: object,
    output_path: Optional[Path],
    status: str,
    message: str,
) -> None:
    report_rows.append(
        {
            "workbook_patient_id": workbook_patient_id,
            "patient_id": patient_id,
            "anchor_uid": anchor_uid,
            "anchor_series_description": anchor_series_description,
            "exam_date": exam_date,
            "study_description": study_description,
            "exam_dir": str(exam_dir) if exam_dir is not None else "",
            "selected_series_descriptions": " | ".join(selected_series_descriptions),
            "selected_series_dirs": " | ".join(str(path) for path in selected_series_dirs),
            "selection_note": selection_note,
            "selection_score": selection_score,
            "phase_source_kind": phase_source_kind,
            "phase_index": phase_index,
            "num_selected_series": num_selected_series,
            "num_dicom_files": num_dicom_files,
            "num_phases": num_phases,
            "num_slices_in_phase": num_slices_in_phase,
            "output_path": str(output_path) if output_path is not None else "",
            "status": status,
            "message": message,
        }
    )


def build_metadata_candidates(csv_path: Path) -> Tuple[Dict[str, List[SeriesCandidate]], Dict[str, SeriesCandidate]]:
    csv_df = pd.read_csv(csv_path)

    csv_df["Series UID"] = csv_df["Series UID"].astype(str).str.strip()
    csv_df["Subject ID"] = csv_df["Subject ID"].astype(str).str.strip()
    csv_df["Study Description"] = csv_df["Study Description"].astype(str).str.strip()
    csv_df["Study Date"] = csv_df["Study Date"].astype(str).str.strip()
    csv_df["Series Description"] = csv_df["Series Description"].astype(str).str.strip()
    csv_df["File Location"] = csv_df["File Location"].astype(str).str.strip()
    csv_df["Modality"] = csv_df["Modality"].astype(str).str.strip()

    by_patient: Dict[str, List[SeriesCandidate]] = defaultdict(list)
    by_uid: Dict[str, SeriesCandidate] = {}

    for _, row in csv_df.iterrows():
        series_dir = (csv_path.parent / row["File Location"]).resolve()
        exam_dir = series_dir.parent
        exam_key = str(exam_dir)
        series_description = str(row["Series Description"]).strip()
        family_key = series_family_key(series_description)
        family_tokens = tokenize_series_family(series_description)

        try:
            num_images = int(row["Number of Images"])
        except Exception:
            num_images = None

        candidate = SeriesCandidate(
            patient_id=str(row["Subject ID"]).strip(),
            exam_key=exam_key,
            exam_dir=exam_dir,
            study_date=str(row["Study Date"]).strip(),
            study_description=str(row["Study Description"]).strip(),
            series_uid=str(row["Series UID"]).strip(),
            series_dir=series_dir,
            series_description=series_description,
            modality=str(row["Modality"]).strip(),
            family_key=family_key,
            family_tokens=family_tokens,
            num_images=num_images,
        )

        by_patient[candidate.patient_id].append(candidate)
        by_uid[candidate.series_uid] = candidate

    return by_patient, by_uid


def select_candidates_for_exam(
    exam_candidates: Sequence[SeriesCandidate],
    anchor_candidate: SeriesCandidate,
    is_anchor_exam: bool,
    min_similarity: float,
) -> Tuple[List[SeriesCandidate], str, float, str]:
    usable = []
    for candidate in exam_candidates:
        if candidate.modality != "MR":
            continue
        if not candidate.series_dir.exists() or not candidate.series_dir.is_dir():
            continue
        usable.append(candidate)

    if not usable:
        return [], "no_usable_mr_series", 0.0, ""

    groups: Dict[str, List[SeriesCandidate]] = defaultdict(list)
    for candidate in usable:
        group_key = candidate.family_key or candidate.series_description.lower()
        groups[group_key].append(candidate)

    anchor_key = anchor_candidate.family_key or anchor_candidate.series_description.lower()
    anchor_tokens = anchor_candidate.family_tokens

    scored_groups = []
    for group_key, candidates in groups.items():
        group_tokens: Set[str] = set()
        for candidate in candidates:
            group_tokens.update(candidate.family_tokens)

        score = family_similarity(anchor_tokens, group_tokens)
        if group_key == anchor_key:
            score += 1.0

        if is_anchor_exam and any(candidate.series_uid == anchor_candidate.series_uid for candidate in candidates):
            score += 2.0

        scored_groups.append((score, group_key, candidates))

    scored_groups.sort(key=lambda item: (-item[0], item[1]))
    best_score, best_group_key, best_candidates = scored_groups[0]

    exact_anchor_group = groups.get(anchor_key)
    if exact_anchor_group is not None and (is_anchor_exam or best_score < 1.0):
        best_group_key = anchor_key
        best_candidates = exact_anchor_group
        best_score = max(best_score, 1.0 if not is_anchor_exam else 2.0)

    if best_score < min_similarity and not is_anchor_exam:
        return [], f"no_family_match_above_threshold:{best_score:.3f}", best_score, best_group_key

    note = f"selected_family={best_group_key}; candidates={len(best_candidates)}; score={best_score:.3f}"
    if len(scored_groups) > 1:
        second_score = scored_groups[1][0]
        if abs(best_score - second_score) < 0.05:
            note += f"; second_best_score={second_score:.3f}"

    return sort_candidates_by_phase(best_candidates), note, best_score, best_group_key


def group_patient_exams(patient_candidates: Sequence[SeriesCandidate]) -> Dict[str, List[SeriesCandidate]]:
    exams: Dict[str, List[SeriesCandidate]] = defaultdict(list)
    for candidate in patient_candidates:
        exams[candidate.exam_key].append(candidate)
    return exams


def reconstruct_exam_selection(
    selection: ExamSelection,
    *,
    workbook_patient_id: str,
    patient_id: str,
    anchor_uid: str,
    anchor_series_description: str,
    output_root: Path,
    phase_gap: int,
    dry_run: bool,
    report_rows: List[Dict[str, object]],
) -> int:
    ordered_candidates = sort_candidates_by_phase(selection.candidates)
    phase_source_kind = "multi_series_group" if len(ordered_candidates) > 1 else "single_series_as_is"

    if not ordered_candidates:
        add_report_row(
            report_rows,
            workbook_patient_id=workbook_patient_id,
            patient_id=patient_id,
            anchor_uid=anchor_uid,
            anchor_series_description=anchor_series_description,
            exam_date=selection.exam_date,
            study_description=selection.study_description,
            exam_dir=selection.exam_dir,
            selected_series_descriptions=[],
            selected_series_dirs=[],
            selection_note=selection.note,
            selection_score=selection.selected_score,
            phase_source_kind=phase_source_kind,
            phase_index="",
            num_selected_series=0,
            num_dicom_files=0,
            num_phases=0,
            num_slices_in_phase=0,
            output_path=None,
            status="no_selected_series",
            message="No series were selected for this exam",
        )
        return 0

    phases: List[List[Path]]
    if len(ordered_candidates) == 1:
        dicom_files = list_dicom_files_sorted(ordered_candidates[0].series_dir)
        if not dicom_files:
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_series_description,
                exam_date=selection.exam_date,
                study_description=selection.study_description,
                exam_dir=selection.exam_dir,
                selected_series_descriptions=[ordered_candidates[0].series_description],
                selected_series_dirs=[ordered_candidates[0].series_dir],
                selection_note=selection.note,
                selection_score=selection.selected_score,
                phase_source_kind=phase_source_kind,
                phase_index="",
                num_selected_series=1,
                num_dicom_files=0,
                num_phases=0,
                num_slices_in_phase=0,
                output_path=None,
                status="empty_selected_series",
                message="Selected series directory contains no files",
            )
            return 0

        phases, phase_source_kind = split_single_series_into_phases(dicom_files, phase_gap=phase_gap)
    else:
        phases = []
        candidate_source_kinds = []
        any_internal_split = False
        for candidate in ordered_candidates:
            dicom_files = list_dicom_files_sorted(candidate.series_dir)
            if not dicom_files:
                continue

            candidate_phases, candidate_source_kind = split_single_series_into_phases(dicom_files, phase_gap=phase_gap)
            if not candidate_phases:
                continue

            phases.extend(candidate_phases)
            candidate_source_kinds.append(candidate_source_kind)
            if len(candidate_phases) > 1:
                any_internal_split = True

        if not phases:
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_series_description,
                exam_date=selection.exam_date,
                study_description=selection.study_description,
                exam_dir=selection.exam_dir,
                selected_series_descriptions=[candidate.series_description for candidate in ordered_candidates],
                selected_series_dirs=[candidate.series_dir for candidate in ordered_candidates],
                selection_note=selection.note,
                selection_score=selection.selected_score,
                phase_source_kind=phase_source_kind,
                phase_index="",
                num_selected_series=len(ordered_candidates),
                num_dicom_files=0,
                num_phases=0,
                num_slices_in_phase=0,
                output_path=None,
                status="empty_selected_group",
                message="Selected phase directories contain no files",
            )
            return 0

        unique_source_kinds = sorted(set(candidate_source_kinds))
        if any_internal_split:
            phase_source_kind = "multi_series_group_with_internal_splits:" + ",".join(unique_source_kinds)
        else:
            phase_source_kind = "multi_series_group"

    patient_out_dir = output_root / patient_id / selection.exam_date
    total_written = 0
    total_dicom_files = sum(len(phase) for phase in phases)
    selected_series_descriptions = [candidate.series_description for candidate in ordered_candidates]
    selected_series_dirs = [candidate.series_dir for candidate in ordered_candidates]

    print(f"[RECONSTRUCT] patient={patient_id} exam={selection.study_description} date={selection.exam_date}")
    print(f"  Selection: {selection.note}")
    for candidate in ordered_candidates:
        print(f"  Series: {candidate.series_description} -> {candidate.series_dir}")
    print(f"  Phase source kind: {phase_source_kind}")
    print(f"  Phases found: {len(phases)}")

    for phase_index, phase_files in enumerate(phases):
        output_path = patient_out_dir / f"volume_phase{phase_index}.nii.gz"
        print(f"  -> phase {phase_index}: {len(phase_files)} slices -> {output_path}")

        if dry_run:
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_series_description,
                exam_date=selection.exam_date,
                study_description=selection.study_description,
                exam_dir=selection.exam_dir,
                selected_series_descriptions=selected_series_descriptions,
                selected_series_dirs=selected_series_dirs,
                selection_note=selection.note,
                selection_score=selection.selected_score,
                phase_source_kind=phase_source_kind,
                phase_index=phase_index,
                num_selected_series=len(ordered_candidates),
                num_dicom_files=total_dicom_files,
                num_phases=len(phases),
                num_slices_in_phase=len(phase_files),
                output_path=output_path,
                status="dry_run",
                message="Dry run enabled; volume not written",
            )
            continue

        try:
            patient_out_dir.mkdir(parents=True, exist_ok=True)
            image = build_sitk_image_from_phase_files(phase_files)
            sitk.WriteImage(image, str(output_path))
            total_written += 1

            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_series_description,
                exam_date=selection.exam_date,
                study_description=selection.study_description,
                exam_dir=selection.exam_dir,
                selected_series_descriptions=selected_series_descriptions,
                selected_series_dirs=selected_series_dirs,
                selection_note=selection.note,
                selection_score=selection.selected_score,
                phase_source_kind=phase_source_kind,
                phase_index=phase_index,
                num_selected_series=len(ordered_candidates),
                num_dicom_files=total_dicom_files,
                num_phases=len(phases),
                num_slices_in_phase=len(phase_files),
                output_path=output_path,
                status="written",
                message="NIfTI volume written successfully",
            )
        except Exception as exc:
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_series_description,
                exam_date=selection.exam_date,
                study_description=selection.study_description,
                exam_dir=selection.exam_dir,
                selected_series_descriptions=selected_series_descriptions,
                selected_series_dirs=selected_series_dirs,
                selection_note=selection.note,
                selection_score=selection.selected_score,
                phase_source_kind=phase_source_kind,
                phase_index=phase_index,
                num_selected_series=len(ordered_candidates),
                num_dicom_files=total_dicom_files,
                num_phases=len(phases),
                num_slices_in_phase=len(phase_files),
                output_path=output_path,
                status="write_error",
                message=str(exc),
            )

    return total_written


def process_dataset(
    *,
    excel_path: Path,
    csv_path: Path,
    dataset_value: str,
    output_root: Path,
    report_csv: Path,
    phase_gap: int,
    dry_run: bool,
    patient_ids: Optional[Sequence[str]],
    max_patients: Optional[int],
    min_similarity: float,
) -> None:
    excel_df = pd.read_excel(excel_path, sheet_name="dataset_info")

    excel_df["dataset"] = excel_df["dataset"].astype(str).str.strip()
    excel_df["patient_id"] = excel_df["patient_id"].astype(str).str.strip()
    excel_df["tcia_series_uid"] = excel_df["tcia_series_uid"].astype(str).str.strip()
    excel_df["normalized_patient_id"] = excel_df["patient_id"].map(normalize_ispy2_patient_id)

    filtered_excel = excel_df[excel_df["dataset"] == dataset_value.strip()].copy()
    if patient_ids:
        wanted = {normalize_ispy2_patient_id(patient_id) for patient_id in patient_ids}
        filtered_excel = filtered_excel[filtered_excel["normalized_patient_id"].isin(wanted)].copy()

    if max_patients is not None:
        filtered_excel = filtered_excel.head(max_patients).copy()

    report_rows: List[Dict[str, object]] = []

    if filtered_excel.empty:
        print(f'No rows found in Excel with dataset == "{dataset_value}".')
        pd.DataFrame(report_rows).to_csv(report_csv, index=False)
        print(f"CSV report written to: {report_csv}")
        return

    metadata_by_patient, metadata_by_uid = build_metadata_candidates(csv_path)

    total_patients = 0
    total_exam_matches = 0
    total_written = 0

    for _, row in filtered_excel.iterrows():
        total_patients += 1

        workbook_patient_id = str(row["patient_id"]).strip()
        patient_id = normalize_ispy2_patient_id(workbook_patient_id)
        anchor_uid = clean_uid(row["tcia_series_uid"])

        anchor_candidate = metadata_by_uid.get(anchor_uid)
        if anchor_candidate is None:
            print(f"[NOT FOUND] patient={patient_id} uid={anchor_uid}")
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description="",
                exam_date="unknown_date",
                study_description="",
                exam_dir=None,
                selected_series_descriptions=[],
                selected_series_dirs=[],
                selection_note="anchor_uid_missing_in_metadata",
                selection_score="",
                phase_source_kind="",
                phase_index="",
                num_selected_series=0,
                num_dicom_files=0,
                num_phases=0,
                num_slices_in_phase=0,
                output_path=None,
                status="uid_not_found_in_metadata",
                message="No matching Series UID found in metadata.csv",
            )
            continue

        patient_candidates = metadata_by_patient.get(patient_id, [])
        if not patient_candidates:
            print(f"[PATIENT NOT FOUND] patient={patient_id}")
            add_report_row(
                report_rows,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_candidate.series_description,
                exam_date="unknown_date",
                study_description="",
                exam_dir=None,
                selected_series_descriptions=[],
                selected_series_dirs=[],
                selection_note="patient_missing_in_metadata",
                selection_score="",
                phase_source_kind="",
                phase_index="",
                num_selected_series=0,
                num_dicom_files=0,
                num_phases=0,
                num_slices_in_phase=0,
                output_path=None,
                status="patient_not_found_in_metadata",
                message="No metadata rows found for the normalized patient ID",
            )
            continue

        exam_groups = group_patient_exams(patient_candidates)
        ordered_exams = sorted(
            exam_groups.values(),
            key=lambda candidates: (
                extract_exam_date_from_exam_folder(candidates[0].exam_dir.name) == "unknown_date",
                extract_exam_date_from_exam_folder(candidates[0].exam_dir.name),
                candidates[0].study_description,
            ),
        )

        print(f"[PATIENT] workbook={workbook_patient_id} metadata={patient_id}")
        print(f"  Anchor UID: {anchor_uid}")
        print(f"  Anchor description: {anchor_candidate.series_description}")
        print(f"  Exams available: {len(ordered_exams)}")

        for exam_candidates in ordered_exams:
            exam_dir = exam_candidates[0].exam_dir
            exam_date = extract_exam_date_from_exam_folder(exam_dir.name)
            study_description = exam_candidates[0].study_description
            is_anchor_exam = exam_dir.resolve() == anchor_candidate.exam_dir.resolve()

            selected_candidates, note, score, selected_family_key = select_candidates_for_exam(
                exam_candidates=exam_candidates,
                anchor_candidate=anchor_candidate,
                is_anchor_exam=is_anchor_exam,
                min_similarity=min_similarity,
            )

            if not selected_candidates:
                print(f"[SKIP EXAM] patient={patient_id} exam={study_description} date={exam_date} reason={note}")
                add_report_row(
                    report_rows,
                    workbook_patient_id=workbook_patient_id,
                    patient_id=patient_id,
                    anchor_uid=anchor_uid,
                    anchor_series_description=anchor_candidate.series_description,
                    exam_date=exam_date,
                    study_description=study_description,
                    exam_dir=exam_dir,
                    selected_series_descriptions=[],
                    selected_series_dirs=[],
                    selection_note=note,
                    selection_score=score,
                    phase_source_kind="",
                    phase_index="",
                    num_selected_series=0,
                    num_dicom_files=0,
                    num_phases=0,
                    num_slices_in_phase=0,
                    output_path=None,
                    status="no_matching_family_in_exam",
                    message=f'No dynamic family matched anchor family "{anchor_candidate.family_key}" in this exam',
                )
                continue

            total_exam_matches += 1
            selection = ExamSelection(
                exam_key=exam_candidates[0].exam_key,
                exam_dir=exam_dir,
                exam_date=exam_date,
                study_description=study_description,
                candidates=selected_candidates,
                note=note,
                selected_family_key=selected_family_key,
                selected_score=score,
            )
            total_written += reconstruct_exam_selection(
                selection,
                workbook_patient_id=workbook_patient_id,
                patient_id=patient_id,
                anchor_uid=anchor_uid,
                anchor_series_description=anchor_candidate.series_description,
                output_root=output_root,
                phase_gap=phase_gap,
                dry_run=dry_run,
                report_rows=report_rows,
            )

    report_df = pd.DataFrame(report_rows)
    report_csv.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(report_csv, index=False)

    print("\nDone.")
    print(f"Patients processed: {total_patients}")
    print(f"Exams selected for reconstruction: {total_exam_matches}")
    if dry_run:
        print("Dry run enabled: no files were written.")
    else:
        print(f"Written NIfTI volumes: {total_written}")
    print(f"CSV report written to: {report_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct ISPY-2 DICOM volumes into NIfTI using the workbook as the patient list, "
            "the workbook UID as the anchor dynamic family, and every available metadata exam for "
            "each matched patient."
        )
    )
    parser.add_argument(
        "--excel_path",
        type=Path,
        default=DEFAULT_EXCEL_PATH,
        help="Path to the clinical_and_imaging_info.xlsx workbook.",
    )
    parser.add_argument(
        "--csv_path",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help="Path to the ISPY-2 metadata.csv file.",
    )
    parser.add_argument(
        "--dataset_value",
        type=str,
        default="ISPY2",
        help='Workbook dataset value to filter, e.g. "ISPY2".',
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output folder where patient/date/volume_phaseX.nii.gz will be written.",
    )
    parser.add_argument(
        "--report_csv",
        type=Path,
        default=DEFAULT_REPORT_CSV,
        help="Path to the reconstruction report CSV.",
    )
    parser.add_argument(
        "--phase_gap",
        type=int,
        default=80,
        help=(
            "A new phase starts when AcquisitionTime or fallback ContentTime changes by at least "
            "this amount for single-folder dynamic series."
        ),
    )
    parser.add_argument(
        "--patient_ids",
        nargs="*",
        default=None,
        help=(
            "Optional patient IDs to process, such as ISPY2-100899 or ISPY2_100899. "
            "If omitted, all workbook ISPY2 patients are considered."
        ),
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Optional limit for quick testing.",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.45,
        help="Minimum family similarity score required for non-anchor exams.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, print and report what would be reconstructed without writing any NIfTI files.",
    )

    args = parser.parse_args()

    process_dataset(
        excel_path=args.excel_path,
        csv_path=args.csv_path,
        dataset_value=args.dataset_value,
        output_root=args.output_root,
        report_csv=args.report_csv,
        phase_gap=args.phase_gap,
        dry_run=args.dry_run,
        patient_ids=args.patient_ids,
        max_patients=args.max_patients,
        min_similarity=args.min_similarity,
    )


if __name__ == "__main__":
    main()
