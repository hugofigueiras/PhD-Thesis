#!/usr/bin/env python3
"""Build Stage 2 MAMA-MIA phase-0 ROI manifests from tumor masks.

The classifier consumes crop coordinates from a CSV manifest, so this script
does not write cropped images to disk. It records a bounding-box crop around the
mask plus a configurable physical margin in millimeters.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAMAMIA_ROOT = Path("data/MAMA-MIA")
DEFAULT_OUTPUT_ROOT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "stage2_mamamia_phase0_roi_expert_margin30"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an ROI pCR manifest from original MAMA-MIA phase-0 images "
            "and expert/automatic/predicted tumor masks."
        )
    )
    parser.add_argument("--mamamia-root", type=Path, default=DEFAULT_MAMAMIA_ROOT)
    parser.add_argument("--clinical-info", type=Path, default=None)
    parser.add_argument("--splits", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--phase-index", type=int, default=0)
    parser.add_argument(
        "--mask-source",
        choices=("expert", "automatic", "predicted"),
        default="expert",
        help="Mask family used to define the crop.",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=None,
        help=(
            "Optional mask root. Defaults to MAMA-MIA segmentations/<mask-source> "
            "for expert/automatic. Required for predicted masks unless they are "
            "stored in a MAMA-MIA-like folder."
        ),
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        nargs=3,
        default=(30.0, 30.0, 20.0),
        metavar=("X", "Y", "Z"),
        help="Physical context margin added around the mask bounding box.",
    )
    parser.add_argument(
        "--component-mode",
        choices=("all", "largest"),
        default="all",
        help="Use all positive voxels or only the largest connected component.",
    )
    parser.add_argument(
        "--strict-affine",
        action="store_true",
        help="Reject image/mask pairs whose affines differ beyond tolerance.",
    )
    parser.add_argument("--affine-atol", type=float, default=1e-3)
    parser.add_argument(
        "--fallback-to-whole-image",
        action="store_true",
        help=(
            "If a mask cannot define an ROI, keep the case by using the whole "
            "image crop and recording the fallback reason. This is useful for "
            "deployable predicted-mask evaluation, where failed segmentations "
            "must not remove test patients."
        ),
    )
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_clinical(path: Path) -> pd.DataFrame:
    clinical = pd.read_excel(path, sheet_name="dataset_info")
    if "patient_id" not in clinical or "pcr" not in clinical:
        raise ValueError(f"{path} must contain patient_id and pcr columns")
    clinical = clinical.copy()
    clinical["patient_id"] = clinical["patient_id"].astype(str)
    clinical["pcr"] = pd.to_numeric(clinical["pcr"], errors="coerce")
    return clinical


def load_splits(path: Path) -> pd.DataFrame:
    split_df = pd.read_csv(path)
    records: list[dict[str, str]] = []
    for column, split_name in (("train_split", "train"), ("test_split", "test")):
        if column not in split_df:
            continue
        for patient_id in split_df[column].dropna().astype(str):
            records.append({"patient_id": patient_id, "split": split_name})
    return pd.DataFrame.from_records(records).drop_duplicates("patient_id")


def phase_path(mamamia_root: Path, patient_id: str, phase_index: int) -> Path:
    return mamamia_root / "images" / patient_id / f"{patient_id}_{phase_index:04d}.nii.gz"


def default_mask_root(mamamia_root: Path, mask_source: str) -> Path:
    if mask_source == "predicted":
        return mamamia_root / "segmentations" / "predicted"
    return mamamia_root / "segmentations" / mask_source


def mask_path(mask_root: Path, patient_id: str) -> Path:
    direct = mask_root / f"{patient_id}.nii.gz"
    if direct.exists():
        return direct
    nnunet_style = mask_root / f"{patient_id}.nii.gz"
    return nnunet_style


def largest_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    structure = np.ones((3, 3, 3), dtype=bool)
    labels, num_labels = ndimage.label(mask, structure=structure)
    if num_labels == 0:
        return mask, 0
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(sizes.argmax()), int(num_labels)


def mask_crop_record(
    mask: np.ndarray,
    image_shape: Iterable[int],
    spacing: Iterable[float],
    affine: np.ndarray,
    margin_mm: Iterable[float],
) -> dict[str, object]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("Cannot crop an empty mask")

    shape = np.asarray(tuple(image_shape), dtype=int)
    spacing_arr = np.asarray(tuple(spacing), dtype=float)
    margin_arr = np.asarray(tuple(margin_mm), dtype=float)
    margin_vox = np.ceil(margin_arr / spacing_arr).astype(int)

    bbox_start = coords.min(axis=0)
    bbox_end = coords.max(axis=0) + 1
    crop_start = bbox_start - margin_vox
    crop_end = bbox_end + margin_vox
    clipped_start = np.maximum(crop_start, 0)
    clipped_end = np.minimum(crop_end, shape)
    center = coords.mean(axis=0)

    voxel_volume_ml = float(abs(np.linalg.det(affine[:3, :3])) / 1000.0)
    return {
        "image_shape_x": int(shape[0]),
        "image_shape_y": int(shape[1]),
        "image_shape_z": int(shape[2]),
        "image_spacing_x": float(spacing_arr[0]),
        "image_spacing_y": float(spacing_arr[1]),
        "image_spacing_z": float(spacing_arr[2]),
        "mask_voxels": int(mask.sum()),
        "mask_volume_ml": float(mask.sum() * voxel_volume_ml),
        "center_vox_x": float(center[0]),
        "center_vox_y": float(center[1]),
        "center_vox_z": float(center[2]),
        "bbox_start_x": int(bbox_start[0]),
        "bbox_start_y": int(bbox_start[1]),
        "bbox_start_z": int(bbox_start[2]),
        "bbox_end_x": int(bbox_end[0]),
        "bbox_end_y": int(bbox_end[1]),
        "bbox_end_z": int(bbox_end[2]),
        "margin_mm_x": float(margin_arr[0]),
        "margin_mm_y": float(margin_arr[1]),
        "margin_mm_z": float(margin_arr[2]),
        "margin_vox_x": int(margin_vox[0]),
        "margin_vox_y": int(margin_vox[1]),
        "margin_vox_z": int(margin_vox[2]),
        "crop_start_x": int(crop_start[0]),
        "crop_start_y": int(crop_start[1]),
        "crop_start_z": int(crop_start[2]),
        "crop_end_x": int(crop_end[0]),
        "crop_end_y": int(crop_end[1]),
        "crop_end_z": int(crop_end[2]),
        "crop_clipped_start_x": int(clipped_start[0]),
        "crop_clipped_start_y": int(clipped_start[1]),
        "crop_clipped_start_z": int(clipped_start[2]),
        "crop_clipped_end_x": int(clipped_end[0]),
        "crop_clipped_end_y": int(clipped_end[1]),
        "crop_clipped_end_z": int(clipped_end[2]),
        "crop_needs_padding": bool(np.any(crop_start < 0) or np.any(crop_end > shape)),
    }


def full_image_crop_record(image_path: Path) -> dict[str, object]:
    image = nib.load(str(image_path))
    spacing = image.header.get_zooms()[:3]
    return {
        "image_shape_x": int(image.shape[0]),
        "image_shape_y": int(image.shape[1]),
        "image_shape_z": int(image.shape[2]),
        "image_spacing_x": float(spacing[0]),
        "image_spacing_y": float(spacing[1]),
        "image_spacing_z": float(spacing[2]),
        "center_vox_x": (int(image.shape[0]) - 1) / 2.0,
        "center_vox_y": (int(image.shape[1]) - 1) / 2.0,
        "center_vox_z": (int(image.shape[2]) - 1) / 2.0,
        "crop_start_x": 0,
        "crop_start_y": 0,
        "crop_start_z": 0,
        "crop_end_x": int(image.shape[0]),
        "crop_end_y": int(image.shape[1]),
        "crop_end_z": int(image.shape[2]),
        "crop_clipped_start_x": 0,
        "crop_clipped_start_y": 0,
        "crop_clipped_start_z": 0,
        "crop_clipped_end_x": int(image.shape[0]),
        "crop_clipped_end_y": int(image.shape[1]),
        "crop_clipped_end_z": int(image.shape[2]),
        "crop_needs_padding": False,
    }


def summarize_manifest(manifest: pd.DataFrame, args: argparse.Namespace) -> dict[str, object]:
    usable = manifest[manifest["usable_for_training"]]
    summary: dict[str, object] = {
        "mamamia_root": str(args.mamamia_root.resolve()),
        "phase_index": int(args.phase_index),
        "mask_source": args.mask_source,
        "margin_mm": [float(v) for v in args.margin_mm],
        "component_mode": args.component_mode,
        "rows": int(len(manifest)),
        "usable_rows": int(len(usable)),
        "roi_status_counts": {
            str(k): int(v)
            for k, v in manifest["roi_status"].value_counts(dropna=False).items()
        },
        "usable_split_counts": {
            str(k): int(v) for k, v in usable["split"].value_counts(dropna=False).items()
        },
        "usable_pcr_counts": {
            str(int(k)): int(v)
            for k, v in usable["pcr"].dropna().astype(int).value_counts().sort_index().items()
        },
        "usable_dataset_counts": {
            str(k): int(v) for k, v in usable["dataset"].value_counts(dropna=False).items()
        },
    }
    if not usable.empty and "mask_volume_ml" in usable:
        volume = usable["mask_volume_ml"].dropna()
        summary["mask_volume_ml_median"] = float(volume.median()) if not volume.empty else None
        summary["mask_volume_ml_min"] = float(volume.min()) if not volume.empty else None
        summary["mask_volume_ml_max"] = float(volume.max()) if not volume.empty else None
    return summary


def main() -> None:
    args = parse_args()
    mamamia_root = args.mamamia_root.resolve()
    clinical_info = resolve_path(args.clinical_info or mamamia_root / "clinical_and_imaging_info.xlsx")
    splits = resolve_path(args.splits or mamamia_root / "train_test_splits.csv")
    mask_root = resolve_path(args.mask_root or default_mask_root(mamamia_root, args.mask_source))
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clinical = load_clinical(clinical_info)
    if args.max_rows is not None:
        clinical = clinical.head(args.max_rows).copy()
    split_df = load_splits(splits)
    merged = clinical.merge(split_df, on="patient_id", how="left")

    accepted_status = f"{args.mask_source}_mask_roi"
    fallback_status = f"{args.mask_source}_mask_fallback_whole_image"
    fallbackable_statuses = {
        "missing_mask",
        "empty_mask",
        "mask_image_shape_mismatch",
        "mask_image_affine_mismatch",
        "mask_read_or_geometry_error",
    }
    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        patient_id = str(row["patient_id"])
        image_path = phase_path(mamamia_root, patient_id, args.phase_index)
        current_mask_path = mask_path(mask_root, patient_id)
        record = row.to_dict()
        record.update(
            {
                "phase_index": int(args.phase_index),
                "case_id": f"mamamia__{patient_id}__phase{args.phase_index:04d}",
                "image_path": str(image_path),
                "reconstructed_image_path": str(image_path),
                "mask_path": str(current_mask_path),
                "mask_source": args.mask_source,
                "roi_source": f"{args.mask_source}_mask_bbox_margin",
                "roi_status": accepted_status,
            }
        )

        if not image_path.exists():
            record["roi_status"] = "missing_phase_image"
        elif not current_mask_path.exists():
            record["roi_status"] = "missing_mask"
        elif pd.isna(row.get("pcr")):
            record["roi_status"] = "missing_pcr_label"
        elif pd.isna(row.get("split")):
            record["roi_status"] = "missing_split"

        if record["roi_status"] == accepted_status:
            try:
                image = nib.load(str(image_path))
                mask_img = nib.load(str(current_mask_path))
                mask = np.asanyarray(mask_img.dataobj) > 0
                record["mask_affine_matches_image"] = bool(
                    np.allclose(image.affine, mask_img.affine, atol=args.affine_atol)
                )
                if mask.shape != image.shape:
                    record["roi_status"] = "mask_image_shape_mismatch"
                elif args.strict_affine and not record["mask_affine_matches_image"]:
                    record["roi_status"] = "mask_image_affine_mismatch"
                elif not mask.any():
                    record["roi_status"] = "empty_mask"
                else:
                    if args.component_mode == "largest":
                        mask, num_components = largest_component(mask)
                    else:
                        num_components = np.nan
                    record["mask_num_components"] = num_components
                    record.update(
                        mask_crop_record(
                            mask=mask,
                            image_shape=image.shape,
                            spacing=image.header.get_zooms()[:3],
                            affine=image.affine,
                            margin_mm=args.margin_mm,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - keep manifest construction robust.
                record["roi_status"] = "mask_read_or_geometry_error"
                record["roi_error"] = str(exc)

        if (
            args.fallback_to_whole_image
            and record["roi_status"] in fallbackable_statuses
            and image_path.exists()
            and pd.notna(row.get("pcr"))
            and pd.notna(row.get("split"))
        ):
            record["roi_fallback_reason"] = record["roi_status"]
            record["roi_status"] = fallback_status
            record["roi_source"] = f"{args.mask_source}_mask_failed_then_whole_image"
            try:
                record.update(full_image_crop_record(image_path))
            except Exception as exc:  # noqa: BLE001
                record["roi_status"] = "fallback_image_read_error"
                record["roi_error"] = str(exc)

        record["usable_for_training"] = record["roi_status"] in {
            accepted_status,
            fallback_status,
        }
        records.append(record)

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_root / "roi_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    summary = summarize_manifest(manifest, args)
    summary.update(
        {
            "clinical_info": str(clinical_info),
            "splits": str(splits),
            "mask_root": str(mask_root),
            "manifest": str(manifest_path),
        }
    )
    summary_path = output_root / "roi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote summary:  {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
