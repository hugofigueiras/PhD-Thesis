#!/usr/bin/env python3
"""Evaluate Stage 2 segmentation predictions against held-out MAMA-MIA masks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEG_ROOT = (
    Path(__file__).resolve().parent / "outputs" / "stage2_mamamia_phase0_segmentation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nnU-Net tumor masks for Stage 2.")
    parser.add_argument("--segmentation-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--predictions-root", type=Path, default=None)
    parser.add_argument("--labels-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument(
        "--skip-hd95",
        action="store_true",
        help="Skip Hausdorff-95 computation for faster checks.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def find_dataset_folder(segmentation_root: Path) -> Path:
    raw_root = segmentation_root / "nnUNet_raw"
    candidates = sorted(raw_root.glob("Dataset*_MAMAMIA_Phase0_TumorSeg"))
    if not candidates:
        candidates = sorted(raw_root.glob("Dataset*"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one nnU-Net raw dataset under {raw_root}, found {len(candidates)}"
        )
    return candidates[0]


def load_mask(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    return np.asanyarray(image.dataobj) > 0, image


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    pred_sum = int(pred.sum())
    target_sum = int(target.sum())
    if pred_sum == 0 and target_sum == 0:
        return 1.0
    denom = pred_sum + target_sum
    if denom == 0:
        return 0.0
    return float(2.0 * np.logical_and(pred, target).sum() / denom)


def surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    structure = ndimage.generate_binary_structure(3, 1)
    eroded = ndimage.binary_erosion(mask, structure=structure, border_value=0)
    return np.logical_xor(mask, eroded)


def hd95_mm(pred: np.ndarray, target: np.ndarray, spacing: tuple[float, float, float]) -> float:
    if not pred.any() or not target.any():
        return float("nan")
    pred_surface = surface(pred)
    target_surface = surface(target)
    if not pred_surface.any() or not target_surface.any():
        return float("nan")

    target_distance = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_distance = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)
    distances = np.concatenate(
        [
            target_distance[pred_surface],
            pred_distance[target_surface],
        ]
    )
    if distances.size == 0:
        return float("nan")
    return float(np.percentile(distances, 95))


def metric_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "prediction_missing": int((df["status"] == "missing_prediction").sum()),
        "shape_mismatch": int((df["status"] == "shape_mismatch").sum()),
        "empty_predictions": int((df["pred_voxels"] == 0).sum()),
    }
    valid = df[df["status"] == "ok"]
    out["valid_rows"] = int(len(valid))
    for column in ("dice", "hd95_mm", "volume_ratio_pred_to_gt"):
        if column in valid:
            values = valid[column].replace([np.inf, -np.inf], np.nan).dropna()
            out[f"{column}_mean"] = float(values.mean()) if not values.empty else None
            out[f"{column}_median"] = float(values.median()) if not values.empty else None
    out["lesion_recall"] = (
        float(valid["lesion_detected"].mean()) if not valid.empty else None
    )
    return out


def main() -> None:
    args = parse_args()
    segmentation_root = resolve_path(args.segmentation_root)
    dataset_folder = find_dataset_folder(segmentation_root)
    predictions_root = resolve_path(args.predictions_root or segmentation_root / "predictionsTs")
    labels_root = resolve_path(args.labels_root or dataset_folder / "labelsTs")
    manifest_path = resolve_path(
        args.manifest or dataset_folder / "stage2_segmentation_manifest.csv"
    )

    manifest = pd.read_csv(manifest_path)
    if args.split != "all":
        manifest = manifest[manifest["segmentation_split"] == args.split].copy()
    if args.max_cases is not None:
        manifest = manifest.head(args.max_cases).copy()

    records: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        patient_id = str(row["patient_id"])
        pred_path = predictions_root / f"{patient_id}.nii.gz"
        label_path = labels_root / f"{patient_id}.nii.gz"
        record: dict[str, Any] = {
            "patient_id": patient_id,
            "prediction_path": str(pred_path),
            "label_path": str(label_path),
            "status": "ok",
        }
        if not pred_path.exists():
            record["status"] = "missing_prediction"
            records.append(record)
            continue
        if not label_path.exists():
            record["status"] = "missing_label"
            records.append(record)
            continue

        pred, pred_img = load_mask(pred_path)
        target, target_img = load_mask(label_path)
        if pred.shape != target.shape:
            record.update(
                {
                    "status": "shape_mismatch",
                    "pred_shape": "x".join(map(str, pred.shape)),
                    "target_shape": "x".join(map(str, target.shape)),
                }
            )
            records.append(record)
            continue

        voxel_volume_ml = float(abs(np.linalg.det(target_img.affine[:3, :3])) / 1000.0)
        pred_voxels = int(pred.sum())
        target_voxels = int(target.sum())
        record.update(
            {
                "pred_voxels": pred_voxels,
                "gt_voxels": target_voxels,
                "pred_volume_ml": pred_voxels * voxel_volume_ml,
                "gt_volume_ml": target_voxels * voxel_volume_ml,
                "volume_ratio_pred_to_gt": (
                    pred_voxels / target_voxels if target_voxels else float("nan")
                ),
                "dice": dice_score(pred, target),
                "lesion_detected": bool(np.logical_and(pred, target).any()),
                "affine_matches": bool(np.allclose(pred_img.affine, target_img.affine, atol=1e-3)),
            }
        )
        if not args.skip_hd95:
            record["hd95_mm"] = hd95_mm(pred, target, target_img.header.get_zooms()[:3])
        records.append(record)

    results = pd.DataFrame.from_records(records)
    summary = {
        "segmentation_root": str(segmentation_root),
        "predictions_root": str(predictions_root),
        "labels_root": str(labels_root),
        "manifest": str(manifest_path),
        "split": args.split,
        "metrics": metric_summary(results),
    }

    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_csv is not None:
        output_csv = resolve_path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_csv, index=False)
        print(f"Wrote {output_csv}")
    if args.output_json is not None:
        output_json = resolve_path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
