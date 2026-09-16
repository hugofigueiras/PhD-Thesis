#!/usr/bin/env python3
"""Summarize Benchmarking probe run metrics into a comparison CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent / "outputs" / "summaries" / "probe_run_summary.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect Benchmarking probe metrics.json files into one CSV."
    )
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any provided run directory is missing metrics.json.",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, object]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    return json.loads(metrics_path.read_text())


def flatten_run(run_dir: Path, metrics: dict[str, object]) -> dict[str, object]:
    selected = metrics.get("official_test_metrics_selected_threshold", {})
    fixed = metrics.get("official_test_metrics_0p5", {})
    split_counts = metrics.get("split_counts", {})
    if not isinstance(selected, dict):
        selected = {}
    if not isinstance(fixed, dict):
        fixed = {}
    if not isinstance(split_counts, dict):
        split_counts = {}

    return {
        "run_dir": str(run_dir),
        "probe": metrics.get("probe"),
        "manifest": metrics.get("manifest"),
        "embeddings": metrics.get("embeddings"),
        "usable_flag": metrics.get("usable_flag"),
        "selection_metric": metrics.get("selection_metric"),
        "best_c": metrics.get("best_c"),
        "selected_threshold_from_validation": metrics.get(
            "selected_threshold_from_validation"
        ),
        "num_embedding_features": metrics.get("num_embedding_features"),
        "official_train_final_fit": split_counts.get("official_train_final_fit"),
        "official_test": split_counts.get("official_test"),
        "test_roc_auc": selected.get("roc_auc"),
        "test_average_precision": selected.get("average_precision"),
        "test_balanced_accuracy": selected.get("balanced_accuracy"),
        "test_accuracy": selected.get("accuracy"),
        "test_precision": selected.get("precision"),
        "test_recall_sensitivity": selected.get("recall_sensitivity"),
        "test_specificity": selected.get("specificity"),
        "test_f1": selected.get("f1"),
        "test_tn": selected.get("tn"),
        "test_fp": selected.get("fp"),
        "test_fn": selected.get("fn"),
        "test_tp": selected.get("tp"),
        "test_roc_auc_0p5": fixed.get("roc_auc"),
        "test_average_precision_0p5": fixed.get("average_precision"),
        "test_balanced_accuracy_0p5": fixed.get("balanced_accuracy"),
    }


def main() -> None:
    args = parse_args()
    rows = []
    skipped = []
    for path in args.run_dirs:
        metrics_path = path / "metrics.json"
        if not metrics_path.exists():
            if args.strict:
                raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
            skipped.append(str(path))
            continue
        rows.append(flatten_run(path, load_metrics(path)))

    if not rows:
        raise RuntimeError("No completed probe runs found")

    summary = pd.DataFrame.from_records(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"Wrote summary: {args.output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} run dirs without metrics.json:")
        for path in skipped:
            print(f"  {path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
