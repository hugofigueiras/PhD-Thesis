#!/usr/bin/env python3
"""Summarize pCR training runs for quick baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize pCR classifier run folders.")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def read_test_result(run_dir: Path) -> tuple[str | None, dict]:
    for filename in ("test_evaluation.json", "best_on_test.json", "test_metrics.json"):
        data = read_json(run_dir / filename)
        if data:
            return filename, data
    return None, {}


def summarize_run(run_dir: Path) -> dict[str, object]:
    metrics_path = run_dir / "metrics.csv"
    config = read_json(run_dir / "run_config.json")
    test_result_file, test_result = read_test_result(run_dir)
    row: dict[str, object] = {"run_dir": str(run_dir)}

    split_summary = config.get("split_summary", {})
    args = config.get("args", {})
    for key in ("train_rows", "val_rows", "test_rows", "tabular_dim", "freeze_encoder"):
        row[key] = split_summary.get(key)
    row["random_init_encoder"] = split_summary.get(
        "random_init_encoder", args.get("random_init_encoder", False)
    )
    row["architecture"] = split_summary.get("architecture", args.get("architecture", "nnunet_encoder"))
    row["image_only"] = args.get("image_only")
    row["roi_status"] = " ".join(args.get("roi_status", [])) if isinstance(args.get("roi_status"), list) else args.get("roi_status")
    row["input_shape"] = "x".join(map(str, args.get("input_shape", []))) if args.get("input_shape") else None
    row["resize_mode"] = split_summary.get("resize_mode", args.get("resize_mode", "stretch"))
    row["test_result_file"] = test_result_file

    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        row["epochs"] = int(metrics["epoch"].max()) if "epoch" in metrics else len(metrics)
        if "val_roc_auc" in metrics and metrics["val_roc_auc"].notna().any():
            best_idx = metrics["val_roc_auc"].idxmax()
        elif "val_loss" in metrics and metrics["val_loss"].notna().any():
            best_idx = metrics["val_loss"].idxmin()
        else:
            best_idx = metrics.index[-1]
        best = metrics.loc[best_idx]
        for key in metrics.columns:
            if key == "epoch":
                row["best_epoch"] = int(best[key])
            elif key.startswith("val_") or key.startswith("train_"):
                row[f"best_{key}"] = float(best[key]) if pd.notna(best[key]) else None

    if isinstance(test_result.get("metrics"), dict):
        row["test_checkpoint"] = test_result.get("checkpoint")
        row["test_checkpoint_epoch"] = test_result.get("checkpoint_epoch")
        test_metrics = test_result["metrics"]
    else:
        test_metrics = test_result

    for key, value in test_metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            row[f"test_{key}"] = value
    return row


def main() -> None:
    args = parse_args()
    rows = [summarize_run(run_dir) for run_dir in args.run_dirs]
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output_csv, index=False)
        print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
