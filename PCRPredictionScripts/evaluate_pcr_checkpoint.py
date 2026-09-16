#!/usr/bin/env python3
"""Evaluate a saved pCR classifier checkpoint on a manifest split.

This is mainly for fair comparisons between ROI strategies. For example, evaluate
the whole-image model only on patients that also had accepted nnU-Net tumor crops.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from train_phase0_nnunet_pcr_classifier import (
    DEFAULT_NNUNET_MODEL_FOLDER,
    Phase0ROIDataset,
    build_pcr_model,
    compute_metrics,
    prepare_splits,
    transform_tabular,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved pCR classifier checkpoint.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--roi-status", nargs="+", default=None)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="test")
    parser.add_argument(
        "--patient-ids-from",
        type=Path,
        default=None,
        help="Optional manifest whose patient IDs define the evaluation subset.",
    )
    parser.add_argument("--patient-ids-from-roi-status", nargs="+", default=None)
    parser.add_argument("--patient-ids-from-split", choices=("train", "test", "all"), default="test")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-predictions", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def filter_usable(df: pd.DataFrame, roi_status: list[str] | None) -> pd.DataFrame:
    out = df.copy()
    if "usable_for_training" in out:
        out = out[as_bool(out["usable_for_training"])]
    if roi_status is not None and "roi_status" in out:
        out = out[out["roi_status"].isin(roi_status)]
    out = out[out["pcr"].notna()].copy()
    out["pcr"] = out["pcr"].astype(int)
    return out.reset_index(drop=True)


def select_split(df: pd.DataFrame, split: str, val_frac: float, seed: int) -> pd.DataFrame:
    if split == "all":
        return df.reset_index(drop=True)
    if split == "test":
        return df[df["split"] == "test"].reset_index(drop=True)
    train_df, val_df, _ = prepare_splits(df, val_frac=val_frac, seed=seed)
    return train_df if split == "train" else val_df


def load_patient_subset(
    path: Path,
    roi_status: list[str] | None,
    split: str,
) -> set[str]:
    df = pd.read_csv(path)
    df = filter_usable(df, roi_status)
    if split != "all":
        df = df[df["split"] == split]
    return set(df["patient_id"].astype(str))


def load_checkpoint(run_dir: Path, checkpoint_name: str) -> dict[str, Any]:
    checkpoint_path = run_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_pos_weight(run_dir: Path) -> torch.Tensor | None:
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    counts = config.get("split_summary", {}).get("train_pcr_counts", {})
    pos = float(counts.get("1", 0))
    neg = float(counts.get("0", 0))
    if pos <= 0 or neg <= 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None,
) -> tuple[dict[str, float], pd.DataFrame]:
    model.eval()
    labels: list[float] = []
    logits_out: list[float] = []
    patient_ids: list[str] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            image = batch["image"].to(device, non_blocking=True)
            tabular = batch["tabular"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)
            logits = model(image, tabular if tabular.shape[1] else None)
            loss = criterion(logits, label)

            total_loss += float(loss.detach().cpu()) * int(label.shape[0])
            total_examples += int(label.shape[0])
            labels.extend(label.detach().cpu().numpy().tolist())
            logits_out.extend(logits.detach().cpu().numpy().tolist())
            patient_ids.extend([str(v) for v in batch["patient_id"]])

    metrics = compute_metrics(labels, logits_out)
    metrics["loss"] = total_loss / max(total_examples, 1)
    probs = 1.0 / (1.0 + np.exp(-np.asarray(logits_out, dtype=np.float32)))
    predictions = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "pcr": np.asarray(labels, dtype=int),
            "logit": logits_out,
            "probability": probs,
            "prediction": (probs >= 0.5).astype(int),
        }
    )
    return metrics, predictions


def main() -> None:
    args = parse_args()
    ckpt = load_checkpoint(args.run_dir, args.checkpoint)
    train_args = ckpt.get("args", {})
    tabular_schema = ckpt.get("tabular_schema", [])

    roi_status = args.roi_status
    if roi_status is None:
        roi_status = train_args.get("roi_status")

    manifest = filter_usable(pd.read_csv(args.manifest), roi_status)
    if args.patient_ids_from is not None:
        patient_ids = load_patient_subset(
            args.patient_ids_from,
            args.patient_ids_from_roi_status,
            args.patient_ids_from_split,
        )
        manifest = manifest[manifest["patient_id"].astype(str).isin(patient_ids)].reset_index(drop=True)

    val_frac = float(train_args.get("val_frac", 0.2))
    seed = int(train_args.get("seed", 2026))
    eval_rows = select_split(manifest, args.split, val_frac=val_frac, seed=seed)
    if eval_rows.empty:
        raise ValueError("No rows selected for evaluation")

    input_shape = tuple(int(v) for v in train_args.get("input_shape", (96, 96, 64)))
    resize_mode = str(train_args.get("resize_mode", "stretch"))
    tabular = transform_tabular(eval_rows, tabular_schema)
    loader = DataLoader(
        Phase0ROIDataset(eval_rows, tabular, input_shape, resize_mode=resize_mode),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model_folder = resolve_path(train_args.get("nnunet_model_folder", DEFAULT_NNUNET_MODEL_FOLDER))
    architecture = str(train_args.get("architecture", "nnunet_encoder"))
    model = build_pcr_model(
        architecture=architecture,
        model_folder=model_folder,
        fold=str(train_args.get("fold", "0")),
        checkpoint_name=str(train_args.get("checkpoint_name", "checkpoint_final.pth")),
        input_shape=input_shape,
        tabular_dim=tabular.shape[1],
        hidden_dim=int(train_args.get("hidden_dim", 128)),
        tabular_hidden_dim=int(train_args.get("tabular_hidden_dim", 64)),
        dropout=float(train_args.get("dropout", 0.25)),
        freeze_encoder=bool(train_args.get("freeze_encoder", False)),
        random_init_encoder=bool(train_args.get("random_init_encoder", False)),
    )
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device)
    model.to(device)

    pos_weight = load_pos_weight(args.run_dir)
    if pos_weight is not None:
        pos_weight = pos_weight.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    metrics, predictions = evaluate(model, loader, criterion, device, args.max_batches)

    result = {
        "run_dir": str(args.run_dir),
        "checkpoint": args.checkpoint,
        "manifest": str(args.manifest),
        "roi_status": roi_status,
        "split": args.split,
        "patient_ids_from": str(args.patient_ids_from) if args.patient_ids_from else None,
        "patient_ids_from_roi_status": args.patient_ids_from_roi_status,
        "patient_ids_from_split": args.patient_ids_from_split,
        "rows": int(len(eval_rows)),
        "pcr_counts": {
            str(int(k)): int(v)
            for k, v in eval_rows["pcr"].value_counts().sort_index().items()
        },
        "architecture": architecture,
        "input_shape": input_shape,
        "resize_mode": resize_mode,
        "metrics": metrics,
        "max_batches": args.max_batches,
    }
    print(json.dumps(json_ready(result), indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(json_ready(result), indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output_json}")
    if args.output_predictions is not None:
        args.output_predictions.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output_predictions, index=False)
        print(f"Wrote {args.output_predictions}")


if __name__ == "__main__":
    main()
