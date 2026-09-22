#!/usr/bin/env python3
"""Evaluate the exploratory shortlist within each official-test source dataset."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from bootstrap_benchmark_shortlist import (
    DEFAULT_COMPARISON,
    DEFAULT_OUTPUT_DIR,
    METRICS,
    align_predictions,
    confidence_interval,
    format_interval,
    load_predictions,
    markdown_table,
    metric_values,
    select_exploratory_shortlist,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute class-stratified bootstrap intervals and paired clinical "
            "differences within each source-dataset subgroup."
        )
    )
    parser.add_argument("--comparison-csv", type=Path, default=DEFAULT_COMPARISON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def make_class_stratified_indices(
    reference: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be at least 1")
    classes = sorted(reference["pcr"].unique().tolist())
    if classes != [0, 1]:
        raise ValueError(
            f"A subgroup must contain both pCR classes; observed classes={classes}"
        )

    pieces = []
    labels = reference["pcr"].to_numpy()
    for label in classes:
        positions = np.flatnonzero(labels == label)
        pieces.append(
            rng.choice(positions, size=(n_bootstrap, len(positions)), replace=True)
        )
    return np.concatenate(pieces, axis=1)


def interval_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
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
    return display


def delta_display(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for metric_key, metric_label in METRICS.items():
        display[f"Delta {metric_label}"] = display.apply(
            lambda row: format_interval(
                row[f"delta_{metric_key}"],
                row[f"delta_{metric_key}_ci_low"],
                row[f"delta_{metric_key}_ci_high"],
            ),
            axis=1,
        )
    display["crop"] = display["crop"].fillna("N/A")
    return display


def write_summary(
    intervals: pd.DataFrame,
    deltas: pd.DataFrame,
    output_path: Path,
    n_bootstrap: int,
    seed: int,
    alpha: float,
) -> None:
    composition = (
        intervals[
            ["dataset", "n_test", "n_pcr", "n_non_pcr", "pcr_prevalence"]
        ]
        .drop_duplicates()
        .sort_values("dataset")
        .copy()
    )
    composition["pcr_prevalence"] = composition["pcr_prevalence"].map(
        lambda value: f"{value:.3f}"
    )

    level = 100.0 * (1.0 - alpha)
    lines = [
        "# Foundation-Model Shortlist By Source Dataset",
        "",
        f"Snapshot date: {date.today().isoformat()}",
        "",
        f"Results use {n_bootstrap:,} paired bootstrap resamples per source dataset",
        f"(seed {seed}), stratified by pCR class. Intervals are {level:.0f}%",
        "percentile intervals.",
        "",
        "## Scope",
        "",
        "This is a subgroup robustness analysis of the official test split, not an",
        "external generalization experiment. Each source dataset also contributes",
        "patients to the official training split. The shortlist was selected after",
        "inspecting aggregate test performance, so all intervals are exploratory.",
        "",
        "## Test Composition",
        "",
        markdown_table(
            composition,
            ["dataset", "n_test", "n_pcr", "n_non_pcr", "pcr_prevalence"],
            {
                "dataset": "Dataset",
                "n_test": "N",
                "n_pcr": "pCR",
                "n_non_pcr": "Non-pCR",
                "pcr_prevalence": "pCR prevalence",
            },
        ),
        "",
        "NACT has only three positive test cases; its intervals are especially",
        "unstable and should not be used for model ranking.",
        "",
    ]

    for dataset in sorted(intervals["dataset"].unique()):
        current_intervals = interval_display(
            intervals[intervals["dataset"] == dataset]
        )
        current_deltas = delta_display(
            deltas[
                (deltas["dataset"] == dataset)
                & (deltas["benchmark_group"] == "Image + clinical")
            ]
        )
        n_test = int(current_intervals["n_test"].iloc[0])
        n_pcr = int(current_intervals["n_pcr"].iloc[0])
        lines.extend(
            [
                f"## {dataset}",
                "",
                f"Test patients: {n_test}; pCR cases: {n_pcr}.",
                "",
                markdown_table(
                    current_intervals,
                    [
                        "model",
                        "benchmark_group",
                        "crop",
                        "input",
                        "AUROC",
                        "AP",
                        "Bal Acc",
                    ],
                    {
                        "model": "Model",
                        "benchmark_group": "Group",
                        "crop": "Crop",
                        "input": "Input",
                    },
                ),
                "",
                "### Image Plus Clinical Minus Clinical-Only",
                "",
                markdown_table(
                    current_deltas,
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
            ]
        )

    lines.extend(
        [
            "## Interpretation Guardrails",
            "",
            "- Compare directions and interval widths across cohorts; do not rank",
            "  models from a single small subgroup.",
            "- The same validation-selected threshold is applied to every source",
            "  subgroup, so balanced accuracy also checks threshold transportability.",
            "- A proper cross-cohort generalization claim requires retraining the probe",
            "  while holding an entire source dataset out from all model selection.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be between 0 and 1")

    comparison = pd.read_csv(args.comparison_csv)
    shortlist = select_exploratory_shortlist(comparison)
    clinical_row = shortlist[shortlist["run_name"] == "clinical_logreg"].iloc[0]
    reference = load_predictions(clinical_row["run_dir"])

    aligned_predictions: dict[str, pd.DataFrame] = {}
    for _, row in shortlist.iterrows():
        run_name = str(row["run_name"])
        current = load_predictions(row["run_dir"])
        aligned_predictions[run_name] = align_predictions(
            reference, current, run_name
        )

    datasets = sorted(reference["dataset"].unique().tolist())
    child_seeds = np.random.SeedSequence(args.seed).spawn(len(datasets))
    interval_rows: list[dict[str, object]] = []
    delta_rows: list[dict[str, object]] = []
    baseline_name = "clinical_logreg"

    for dataset, child_seed in zip(datasets, child_seeds):
        mask = reference["dataset"].eq(dataset).to_numpy()
        subgroup_reference = reference.loc[mask].reset_index(drop=True)
        rng = np.random.default_rng(child_seed)
        bootstrap_indices = make_class_stratified_indices(
            subgroup_reference, args.n_bootstrap, rng
        )
        full_indices = np.arange(len(subgroup_reference), dtype=np.int64)

        points: dict[str, np.ndarray] = {}
        replicates: dict[str, np.ndarray] = {}
        for _, row in shortlist.iterrows():
            run_name = str(row["run_name"])
            subgroup = aligned_predictions[run_name].loc[mask].reset_index(drop=True)
            points[run_name] = metric_values(subgroup, full_indices)
            replicates[run_name] = np.vstack(
                [metric_values(subgroup, indices) for indices in bootstrap_indices]
            )

        n_test = len(subgroup_reference)
        n_pcr = int(subgroup_reference["pcr"].sum())
        common = {
            "dataset": dataset,
            "n_test": n_test,
            "n_pcr": n_pcr,
            "n_non_pcr": n_test - n_pcr,
            "pcr_prevalence": n_pcr / n_test,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "stratification": "pcr_within_dataset",
        }

        for _, row in shortlist.iterrows():
            run_name = str(row["run_name"])
            record = {
                **common,
                "model": row["model"],
                "benchmark_group": row["benchmark_group"],
                "crop": row["crop"],
                "input": row["input"],
                "run_name": run_name,
                "run_dir": row["run_dir"],
                "selection_reason": row["selection_reason"],
            }
            for metric_index, metric_key in enumerate(METRICS):
                low, high = confidence_interval(
                    replicates[run_name][:, metric_index], args.alpha
                )
                record[metric_key] = float(points[run_name][metric_index])
                record[f"{metric_key}_ci_low"] = low
                record[f"{metric_key}_ci_high"] = high
            interval_rows.append(record)

            if run_name == baseline_name:
                continue
            point_delta = points[run_name] - points[baseline_name]
            bootstrap_delta = replicates[run_name] - replicates[baseline_name]
            delta_record = {
                **common,
                "model": row["model"],
                "benchmark_group": row["benchmark_group"],
                "crop": row["crop"],
                "input": row["input"],
                "run_name": run_name,
                "baseline_run_name": baseline_name,
            }
            for metric_index, metric_key in enumerate(METRICS):
                values = bootstrap_delta[:, metric_index]
                low, high = confidence_interval(values, args.alpha)
                delta_record[f"delta_{metric_key}"] = float(
                    point_delta[metric_index]
                )
                delta_record[f"delta_{metric_key}_ci_low"] = low
                delta_record[f"delta_{metric_key}_ci_high"] = high
                delta_record[
                    f"bootstrap_probability_delta_gt_zero_{metric_key}"
                ] = float(np.mean(values > 0.0))
            delta_rows.append(delta_record)

    intervals = pd.DataFrame.from_records(interval_rows)
    deltas = pd.DataFrame.from_records(delta_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    intervals_path = args.output_dir / "shortlist_by_dataset_intervals.csv"
    deltas_path = args.output_dir / "shortlist_by_dataset_deltas_vs_clinical.csv"
    summary_path = args.output_dir / "shortlist_by_dataset_summary.md"
    intervals.to_csv(intervals_path, index=False)
    deltas.to_csv(deltas_path, index=False)
    write_summary(
        intervals,
        deltas,
        summary_path,
        args.n_bootstrap,
        args.seed,
        args.alpha,
    )

    print(f"Shortlist runs: {len(shortlist)}")
    print(f"Source datasets: {', '.join(datasets)}")
    print(f"Bootstrap resamples per dataset: {args.n_bootstrap}")
    print(f"Wrote intervals: {intervals_path}")
    print(f"Wrote paired deltas: {deltas_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
