#!/usr/bin/env python3
"""Bootstrap uncertainty for the exploratory foundation-model shortlist."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_COMPARISON = (
    ROOT / "outputs" / "summaries" / "cross_model_first_pass_comparison.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "summaries"

METRICS = {
    "roc_auc": "AUROC",
    "average_precision": "AP",
    "balanced_accuracy": "Bal Acc",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-and-class-stratified bootstrap intervals and paired "
            "differences from the clinical-only baseline."
        )
    )
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def select_exploratory_shortlist(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select clinical plus per-model AUROC and overall metric winners.

    Selection uses the observed test metrics and is therefore explicitly
    exploratory. The bootstrap quantifies sampling uncertainty; it does not
    remove the optimism introduced by test-set selection.
    """

    required = {
        "model",
        "benchmark_group",
        "crop",
        "input",
        "run_name",
        "run_dir",
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
    }
    missing = required - set(comparison.columns)
    if missing:
        raise ValueError(
            f"Comparison table is missing required columns: {sorted(missing)}"
        )

    reasons: dict[str, list[str]] = defaultdict(list)
    selected_rows: dict[str, pd.Series] = {}

    def add(row: pd.Series, reason: str) -> None:
        run_name = str(row["run_name"])
        selected_rows[run_name] = row
        if reason not in reasons[run_name]:
            reasons[run_name].append(reason)

    clinical = comparison[comparison["run_name"] == "clinical_logreg"]
    if len(clinical) != 1:
        raise ValueError(
            "Expected exactly one clinical_logreg row in the comparison table"
        )
    add(clinical.iloc[0], "Clinical-only reference")

    primary = comparison[
        comparison["benchmark_group"].isin(["Image-only", "Image + clinical"])
    ].copy()
    if primary.empty:
        raise ValueError("No primary image benchmark rows were found")

    for (model, group), rows in primary.groupby(
        ["model", "benchmark_group"], sort=True
    ):
        rows = rows.sort_values("run_name")
        best = rows.loc[rows["test_roc_auc"].idxmax()]
        add(best, f"Exploratory per-model {group} AUROC winner")

    metric_columns = {
        "AUROC": "test_roc_auc",
        "AP": "test_average_precision",
        "balanced accuracy": "test_balanced_accuracy",
    }
    for group in ("Image-only", "Image + clinical"):
        rows = primary[primary["benchmark_group"] == group].sort_values("run_name")
        for metric_label, metric_column in metric_columns.items():
            best = rows.loc[rows[metric_column].idxmax()]
            add(best, f"Exploratory overall {group} {metric_label} winner")

    shortlist = pd.DataFrame(selected_rows.values()).copy()
    shortlist["selection_reason"] = shortlist["run_name"].map(
        lambda run_name: "; ".join(reasons[str(run_name)])
    )
    group_order = {"Clinical-only": 0, "Image-only": 1, "Image + clinical": 2}
    shortlist["_group_order"] = shortlist["benchmark_group"].map(group_order)
    shortlist = shortlist.sort_values(
        ["_group_order", "model", "crop", "input", "run_name"]
    ).drop(columns="_group_order")
    return shortlist.reset_index(drop=True)


def load_predictions(run_dir: str | Path) -> pd.DataFrame:
    path = resolve_project_path(run_dir) / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing test predictions: {path}")
    predictions = pd.read_csv(path)
    required = {"patient_id", "dataset", "pcr", "pred_prob", "pred_label"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    predictions = predictions.copy()
    predictions["patient_id"] = predictions["patient_id"].astype(str)
    predictions["dataset"] = predictions["dataset"].astype(str)
    predictions["pcr"] = pd.to_numeric(predictions["pcr"], errors="raise").astype(int)
    predictions["pred_prob"] = pd.to_numeric(
        predictions["pred_prob"], errors="raise"
    ).astype(float)
    predictions["pred_label"] = pd.to_numeric(
        predictions["pred_label"], errors="raise"
    ).astype(int)

    if predictions["patient_id"].duplicated().any():
        duplicates = predictions.loc[
            predictions["patient_id"].duplicated(), "patient_id"
        ].tolist()
        raise ValueError(f"Duplicate patient IDs in {path}: {duplicates[:5]}")
    if not predictions["pcr"].isin([0, 1]).all():
        raise ValueError(f"Non-binary pCR labels in {path}")
    if not predictions["pred_prob"].between(0.0, 1.0).all():
        raise ValueError(f"Predicted probabilities outside [0, 1] in {path}")
    if not predictions["pred_label"].isin([0, 1]).all():
        raise ValueError(f"Non-binary predicted labels in {path}")
    return predictions.sort_values("patient_id").reset_index(drop=True)


def align_predictions(
    reference: pd.DataFrame, current: pd.DataFrame, run_name: str
) -> pd.DataFrame:
    reference_ids = set(reference["patient_id"])
    current_ids = set(current["patient_id"])
    if reference_ids != current_ids:
        missing = sorted(reference_ids - current_ids)
        extra = sorted(current_ids - reference_ids)
        raise ValueError(
            f"{run_name} does not use the clinical reference cohort; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    aligned = current.set_index("patient_id").loc[reference["patient_id"]].reset_index()
    if not np.array_equal(aligned["pcr"].to_numpy(), reference["pcr"].to_numpy()):
        raise ValueError(f"pCR labels differ from the reference for {run_name}")
    if not np.array_equal(
        aligned["dataset"].to_numpy(), reference["dataset"].to_numpy()
    ):
        raise ValueError(f"Dataset labels differ from the reference for {run_name}")
    return aligned


def metric_values(predictions: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    y_true = predictions["pcr"].to_numpy()[indices]
    y_prob = predictions["pred_prob"].to_numpy()[indices]
    y_pred = predictions["pred_label"].to_numpy()[indices]
    return np.asarray(
        [
            roc_auc_score(y_true, y_prob),
            average_precision_score(y_true, y_prob),
            balanced_accuracy_score(y_true, y_pred),
        ],
        dtype=float,
    )


def make_stratified_bootstrap_indices(
    reference: pd.DataFrame, n_bootstrap: int, seed: int
) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1")
    rng = np.random.default_rng(seed)
    pieces = []
    for _, positions in reference.groupby(["dataset", "pcr"], sort=True).indices.items():
        positions = np.asarray(positions, dtype=np.int64)
        pieces.append(
            rng.choice(positions, size=(n_bootstrap, len(positions)), replace=True)
        )
    return np.concatenate(pieces, axis=1)


def confidence_interval(
    values: np.ndarray, alpha: float
) -> tuple[float, float]:
    return (
        float(np.quantile(values, alpha / 2.0)),
        float(np.quantile(values, 1.0 - alpha / 2.0)),
    )


def markdown_table(
    frame: pd.DataFrame, columns: list[str], labels: dict[str, str]
) -> str:
    header = "| " + " | ".join(labels.get(column, column) for column in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = [header, separator]
    for _, row in frame.iterrows():
        values = [str(row[column]).replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def format_interval(point: float, low: float, high: float) -> str:
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def write_markdown_summary(
    intervals: pd.DataFrame,
    deltas: pd.DataFrame,
    output_path: Path,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> None:
    display = intervals.copy()
    for metric_key, metric_label in METRICS.items():
        display[metric_label] = display.apply(
            lambda row: format_interval(
                row[metric_key],
                row[f"{metric_key}_ci_low"],
                row[f"{metric_key}_ci_high"],
            ),
            axis=1,
        )
    display["crop"] = display["crop"].fillna("N/A")

    multimodal_deltas = deltas[
        deltas["benchmark_group"] == "Image + clinical"
    ].copy()
    for metric_key, metric_label in METRICS.items():
        multimodal_deltas[f"Delta {metric_label}"] = multimodal_deltas.apply(
            lambda row: format_interval(
                row[f"delta_{metric_key}"],
                row[f"delta_{metric_key}_ci_low"],
                row[f"delta_{metric_key}_ci_high"],
            ),
            axis=1,
        )

    level = 100.0 * (1.0 - alpha)
    lines = [
        "# Foundation-Model Shortlist Bootstrap Summary",
        "",
        f"Snapshot date: {date.today().isoformat()}",
        "",
        f"Intervals use {n_bootstrap:,} paired bootstrap resamples (seed {seed})",
        "stratified jointly by source dataset and pCR class. Balanced accuracy",
        "uses each probe's threshold selected on its training-only validation split.",
        "",
        "## Important Scope",
        "",
        "The shortlist was selected after inspecting official-test performance. These",
        f"{level:.0f}% intervals are descriptive uncertainty estimates, not confirmatory",
        "post-selection significance tests. They do not correct for trying multiple",
        "models, inputs, crops, or fusion variants.",
        "",
        "## Shortlist Intervals",
        "",
        markdown_table(
            display,
            ["model", "benchmark_group", "crop", "input", "AUROC", "AP", "Bal Acc"],
            {
                "model": "Model",
                "benchmark_group": "Group",
                "crop": "Crop",
                "input": "Input",
            },
        ),
        "",
        "## Image Plus Clinical Minus Clinical-Only",
        "",
        "Positive values favor image plus clinical. Every difference uses the same",
        "resampled patients for both models.",
        "",
        markdown_table(
            multimodal_deltas,
            [
                "model",
                "crop",
                "input",
                "Delta AUROC",
                "Delta AP",
                "Delta Bal Acc",
            ],
            {"model": "Model", "crop": "Crop", "input": "Input"},
        ),
        "",
        "## Interpretation Guardrails",
        "",
        "- An interval crossing zero means the paired bootstrap does not show a stable",
        "  direction of difference at this sample size.",
        "- An interval excluding zero is still exploratory because configurations were",
        "  compared and shortlisted using this same test set.",
        "- Final model selection or tuning should return to training-only nested",
        "  validation or use a genuinely untouched external cohort.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    comparison = pd.read_csv(args.comparison_csv)
    shortlist = select_exploratory_shortlist(comparison)

    clinical_row = shortlist[shortlist["run_name"] == "clinical_logreg"].iloc[0]
    reference = load_predictions(clinical_row["run_dir"])
    bootstrap_indices = make_stratified_bootstrap_indices(
        reference, args.n_bootstrap, args.seed
    )
    full_indices = np.arange(len(reference), dtype=np.int64)

    points: dict[str, np.ndarray] = {}
    replicates: dict[str, np.ndarray] = {}
    aligned_predictions: dict[str, pd.DataFrame] = {}

    for _, row in shortlist.iterrows():
        run_name = str(row["run_name"])
        current = load_predictions(row["run_dir"])
        aligned = align_predictions(reference, current, run_name)
        aligned_predictions[run_name] = aligned
        points[run_name] = metric_values(aligned, full_indices)
        replicates[run_name] = np.vstack(
            [metric_values(aligned, indices) for indices in bootstrap_indices]
        )

        expected = np.asarray(
            [
                row["test_roc_auc"],
                row["test_average_precision"],
                row["test_balanced_accuracy"],
            ],
            dtype=float,
        )
        if not np.allclose(points[run_name], expected, rtol=0.0, atol=1e-10):
            raise ValueError(
                f"Saved predictions and comparison metrics disagree for {run_name}: "
                f"{points[run_name]} versus {expected}"
            )

    interval_rows = []
    for _, row in shortlist.iterrows():
        run_name = str(row["run_name"])
        record = {
            "model": row["model"],
            "benchmark_group": row["benchmark_group"],
            "crop": row["crop"],
            "input": row["input"],
            "run_name": run_name,
            "run_dir": row["run_dir"],
            "selection_reason": row["selection_reason"],
            "n_test": len(reference),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "stratification": "dataset+pcr",
        }
        for metric_index, metric_key in enumerate(METRICS):
            low, high = confidence_interval(
                replicates[run_name][:, metric_index], args.alpha
            )
            record[metric_key] = float(points[run_name][metric_index])
            record[f"{metric_key}_ci_low"] = low
            record[f"{metric_key}_ci_high"] = high
        interval_rows.append(record)
    intervals = pd.DataFrame.from_records(interval_rows)

    baseline_name = "clinical_logreg"
    delta_rows = []
    for _, row in shortlist.iterrows():
        run_name = str(row["run_name"])
        if run_name == baseline_name:
            continue
        record = {
            "model": row["model"],
            "benchmark_group": row["benchmark_group"],
            "crop": row["crop"],
            "input": row["input"],
            "run_name": run_name,
            "baseline_run_name": baseline_name,
            "n_test": len(reference),
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "stratification": "dataset+pcr",
        }
        point_delta = points[run_name] - points[baseline_name]
        bootstrap_delta = replicates[run_name] - replicates[baseline_name]
        for metric_index, metric_key in enumerate(METRICS):
            values = bootstrap_delta[:, metric_index]
            low, high = confidence_interval(values, args.alpha)
            record[f"delta_{metric_key}"] = float(point_delta[metric_index])
            record[f"delta_{metric_key}_ci_low"] = low
            record[f"delta_{metric_key}_ci_high"] = high
            record[f"bootstrap_probability_delta_gt_zero_{metric_key}"] = float(
                np.mean(values > 0.0)
            )
        delta_rows.append(record)
    deltas = pd.DataFrame.from_records(delta_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals_path = args.output_dir / "bootstrap_shortlist_intervals.csv"
    deltas_path = args.output_dir / "bootstrap_shortlist_deltas_vs_clinical.csv"
    summary_path = args.output_dir / "bootstrap_shortlist_summary.md"
    intervals.to_csv(intervals_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    write_markdown_summary(
        intervals,
        deltas,
        summary_path,
        args.n_bootstrap,
        args.seed,
        args.alpha,
    )

    print(f"Shortlist runs: {len(shortlist)}")
    print(f"Official test patients: {len(reference)}")
    print(f"Bootstrap resamples: {args.n_bootstrap}")
    print(f"Wrote intervals: {intervals_path}")
    print(f"Wrote paired deltas: {deltas_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
