from __future__ import annotations

import re
import shutil
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECONSTRUCTED_ROOT = PROJECT_ROOT / "Reconstructed_Datasets"
OUTPUT_ROOT = RECONSTRUCTED_ROOT / "Original-VOIs"
DATASET_DIRS = ("ISPY1-Volumes", "ISPY2-Volumes", "NACT-Volumes")

ORIGINAL_ROOTS = {
    "ISPY1-Volumes": Path(
        "data/ISPY-1/manifest-PyHQgfru6393647793776378748/ISPY1"
    ),
    "ISPY2-Volumes": Path(
        "data/ISPY-2/manifest-1641168072464/ISPY2"
    ),
    "NACT-Volumes": Path(
        "data/Breast MRI NACT Pilot/"
        "manifest-RbPGRCVv7392292744865323559/Breast-MRI-NACT-Pilot"
    ),
}


def normalize_source_patient_id(dataset_dir_name: str, reconstructed_patient_id: str) -> str:
    if dataset_dir_name == "NACT-Volumes":
        suffix = reconstructed_patient_id.split("_", 1)[1]
        return f"UCSF-BR-{suffix.zfill(2)}"
    return reconstructed_patient_id


def load_reconstructed_dates() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for dataset_dir_name in DATASET_DIRS:
        dataset_dir = RECONSTRUCTED_ROOT / dataset_dir_name
        for patient_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
            for date_dir in sorted(path for path in patient_dir.iterdir() if path.is_dir()):
                rows.append(
                    {
                        "dataset_dir_name": dataset_dir_name,
                        "reconstructed_patient_id": patient_dir.name,
                        "acquisition_date": date_dir.name,
                    }
                )
    return rows


def describe_series(series_dir: Path) -> str:
    parts = series_dir.name.split("-")
    if len(parts) >= 3:
        return "-".join(parts[1:-1])
    return series_dir.name


def slugify(text: str) -> str:
    text = text.lower().replace("&", "and")
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def list_matching_exams(patient_root: Path, acquisition_date: str) -> list[Path]:
    return sorted(
        path
        for path in patient_root.glob(f"{acquisition_date}-*")
        if path.is_dir()
    )


def find_series(exam_dir: Path, required_terms: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for series_dir in exam_dir.iterdir():
        if not series_dir.is_dir():
            continue
        series_name = series_dir.name.lower()
        if all(term in series_name for term in required_terms):
            matches.append(series_dir)
    return sorted(matches)


def choose_series_for_exam(dataset_dir_name: str, exam_dir: Path) -> list[dict[str, object]]:
    if dataset_dir_name == "ISPY2-Volumes":
        analysis_masks = find_series(exam_dir, ("analysis mask",))
        if len(analysis_masks) != 1:
            return []
        source = analysis_masks[0]
        return [
            {
                "selection_rule": "analysis_mask",
                "target_series_dir_name": slugify(describe_series(source)),
                "source_dir": source,
            }
        ]

    if dataset_dir_name == "NACT-Volumes":
        pe_segmentations = find_series(exam_dir, ("pe segmentation",))
        breast_segmentations = find_series(exam_dir, ("breast tissue segmentation",))
        if len(pe_segmentations) != 1 or len(breast_segmentations) != 1:
            return []
        return [
            {
                "selection_rule": "paired_segmentation",
                "target_series_dir_name": slugify(describe_series(pe_segmentations[0])),
                "source_dir": pe_segmentations[0],
            },
            {
                "selection_rule": "paired_segmentation",
                "target_series_dir_name": slugify(describe_series(breast_segmentations[0])),
                "source_dir": breast_segmentations[0],
            },
        ]

    voi_pe_segmentations = find_series(exam_dir, ("voi pe segmentation",))
    voi_breast_segmentations = find_series(
        exam_dir, ("voi breast tissue segmentation",)
    )
    if len(voi_pe_segmentations) == 1 and len(voi_breast_segmentations) == 1:
        return [
            {
                "selection_rule": "voi_pair",
                "target_series_dir_name": slugify(describe_series(voi_pe_segmentations[0])),
                "source_dir": voi_pe_segmentations[0],
            },
            {
                "selection_rule": "voi_pair",
                "target_series_dir_name": slugify(describe_series(voi_breast_segmentations[0])),
                "source_dir": voi_breast_segmentations[0],
            },
        ]

    pe_segmentations = find_series(exam_dir, ("pe segmentation",))
    breast_segmentations = find_series(exam_dir, ("breast tissue segmentation",))
    if len(pe_segmentations) == 1 and len(breast_segmentations) == 1:
        return [
            {
                "selection_rule": "segmentation_pair_fallback",
                "target_series_dir_name": slugify(describe_series(pe_segmentations[0])),
                "source_dir": pe_segmentations[0],
            },
            {
                "selection_rule": "segmentation_pair_fallback",
                "target_series_dir_name": slugify(describe_series(breast_segmentations[0])),
                "source_dir": breast_segmentations[0],
            },
        ]

    return []


def choose_exam_and_series(
    dataset_dir_name: str,
    exams: list[Path],
) -> tuple[Path | None, list[dict[str, object]]]:
    candidates: list[tuple[int, Path, list[dict[str, object]]]] = []
    for exam_dir in exams:
        series = choose_series_for_exam(dataset_dir_name, exam_dir)
        if not series:
            continue

        selection_rule = str(series[0]["selection_rule"])
        score = {
            "voi_pair": 3,
            "paired_segmentation": 2,
            "analysis_mask": 2,
            "segmentation_pair_fallback": 1,
        }[selection_rule]
        candidates.append((score, exam_dir, series))

    if not candidates:
        return None, []

    best_score = max(candidate[0] for candidate in candidates)
    best_candidates = [candidate for candidate in candidates if candidate[0] == best_score]
    if len(best_candidates) != 1:
        candidate_names = ", ".join(candidate[1].name for candidate in best_candidates)
        raise ValueError(
            f"Ambiguous exam selection for {dataset_dir_name}: {candidate_names}"
        )

    _, exam_dir, series = best_candidates[0]
    return exam_dir, series


def copy_series_dir(source_dir: Path, destination_dir: Path) -> int:
    shutil.copytree(source_dir, destination_dir)
    return sum(1 for path in destination_dir.rglob("*") if path.is_file())


def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    copied_dates = 0
    missing_dates = 0
    copied_series = 0
    copied_files = 0

    for row in load_reconstructed_dates():
        dataset_dir_name = row["dataset_dir_name"]
        reconstructed_patient_id = row["reconstructed_patient_id"]
        acquisition_date = row["acquisition_date"]
        source_patient_id = normalize_source_patient_id(
            dataset_dir_name=dataset_dir_name,
            reconstructed_patient_id=reconstructed_patient_id,
        )
        patient_root = ORIGINAL_ROOTS[dataset_dir_name] / source_patient_id

        if not patient_root.exists():
            raise FileNotFoundError(f"Missing source patient folder: {patient_root}")

        exams = list_matching_exams(patient_root, acquisition_date)
        if not exams:
            manifest_rows.append(
                {
                    "dataset_dir_name": dataset_dir_name,
                    "reconstructed_patient_id": reconstructed_patient_id,
                    "source_patient_id": source_patient_id,
                    "acquisition_date": acquisition_date,
                    "status": "missing_exam",
                    "selection_rule": "",
                    "source_exam_dir": "",
                    "source_series_dir": "",
                    "target_series_dir": "",
                    "copied_file_count": 0,
                }
            )
            missing_dates += 1
            continue

        exam_dir, selected_series = choose_exam_and_series(dataset_dir_name, exams)
        if exam_dir is None or not selected_series:
            manifest_rows.append(
                {
                    "dataset_dir_name": dataset_dir_name,
                    "reconstructed_patient_id": reconstructed_patient_id,
                    "source_patient_id": source_patient_id,
                    "acquisition_date": acquisition_date,
                    "status": "missing_source_voi",
                    "selection_rule": "",
                    "source_exam_dir": "|".join(str(path) for path in exams),
                    "source_series_dir": "",
                    "target_series_dir": "",
                    "copied_file_count": 0,
                }
            )
            missing_dates += 1
            continue

        target_date_dir = OUTPUT_ROOT / reconstructed_patient_id / acquisition_date
        target_date_dir.mkdir(parents=True, exist_ok=True)
        copied_dates += 1

        for series in selected_series:
            source_dir = Path(series["source_dir"])
            target_series_dir = target_date_dir / str(series["target_series_dir_name"])
            file_count = copy_series_dir(source_dir, target_series_dir)

            copied_series += 1
            copied_files += file_count
            manifest_rows.append(
                {
                    "dataset_dir_name": dataset_dir_name,
                    "reconstructed_patient_id": reconstructed_patient_id,
                    "source_patient_id": source_patient_id,
                    "acquisition_date": acquisition_date,
                    "status": "copied",
                    "selection_rule": str(series["selection_rule"]),
                    "source_exam_dir": str(exam_dir),
                    "source_series_dir": str(source_dir),
                    "target_series_dir": str(target_series_dir),
                    "copied_file_count": file_count,
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUTPUT_ROOT / "manifest.csv", index=False)
    print(
        "Copied "
        f"{copied_series} VOI series across {copied_dates} reconstructed dates "
        f"into {OUTPUT_ROOT}. Missing source VOIs for {missing_dates} dates. "
        f"Total copied files: {copied_files}"
    )


if __name__ == "__main__":
    main()
