#!/usr/bin/env python3
"""Build cross-model first-pass benchmark summaries and figures."""

from __future__ import annotations

import argparse
import os
from datetime import date
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_SUMMARY = ROOT / "outputs" / "summaries" / "probe_run_summary.csv"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "summaries"

MODEL_DISPLAY = {
    "pillar0_breastmri": "Pillar-0 BreastMRI",
    "radiodino": "RadioDINO",
    "biomedclip": "BiomedCLIP",
    "curia": "Curia",
    "medsiglip": "MedSigLIP",
    "radimagenet": "RadImageNet",
    "jolia_cross_modality": "Jolia (CT-to-MRI stress test)",
    "clinical": "Clinical baseline",
}

MODEL_ORDER = {
    "clinical": 0,
    "pillar0_breastmri": 1,
    "radiodino": 2,
    "biomedclip": 3,
    "curia": 4,
    "medsiglip": 5,
    "radimagenet": 6,
    "jolia_cross_modality": 7,
}

INPUT_LABELS = {
    "phase0": "Phase 0",
    "phase1": "Phase 1",
    "phase2": "Phase 2",
    "last_phase": "Last phase",
    "phase1_minus_phase0": "Phase 1 - phase 0",
    "phase2_minus_phase0": "Phase 2 - phase 0",
    "last_phase_minus_phase0": "Last phase - phase 0",
    "raw_phases_fusion": "Raw phase fusion",
    "subtractions_fusion": "Subtraction fusion",
    "all_dce_fusion": "All DCE fusion",
    "selected_fusion": "Selected ROI fusion",
    "phase0_phase1_phase2": "Phase 0 + phase 1 + phase 2",
    "phase0_phase1_last": "Phase 0 + phase 1 + last phase",
    "phase0_phase2_last": "Phase 0 + phase 2 + last phase",
    "subtractions_phase1_phase2_last": "Subtraction triplet",
}

METRICS = {
    "AUROC": "test_roc_auc",
    "AP": "test_average_precision",
    "Bal Acc": "test_balanced_accuracy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cross-model summary tables and plots."
    )
    parser.add_argument("--summary-csv", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def parse_run_name(run_dir: str) -> dict[str, str]:
    name = Path(run_dir).name
    if name == "clinical_logreg":
        return {
            "model_key": "clinical",
            "model": MODEL_DISPLAY["clinical"],
            "input_key": "clinical",
            "input": "Clinical variables",
            "crop": "N/A",
            "feature_set": "Clinical-only",
            "clinical_data": "Yes",
            "benchmark_group": "Clinical-only",
        }

    model_key = ""
    remainder = name
    for candidate in (
        "pillar0_breastmri",
        "radiodino",
        "biomedclip",
        "curia",
        "medsiglip",
        "radimagenet",
        "jolia_cross_modality",
    ):
        prefix = f"{candidate}_"
        if name.startswith(prefix):
            model_key = candidate
            remainder = name[len(prefix) :]
            break
    if not model_key:
        raise ValueError(f"Cannot parse model key from run name: {name}")

    has_clinical = remainder.endswith("_image_clinical_logreg")
    if has_clinical:
        remainder = remainder[: -len("_image_clinical_logreg")]
    elif remainder.endswith("_logreg"):
        remainder = remainder[: -len("_logreg")]
    else:
        raise ValueError(f"Cannot parse probe suffix from run name: {name}")

    if remainder.startswith("whole_"):
        crop = "Whole volume"
        input_key = remainder[len("whole_") :]
    elif remainder.startswith("expert_roi_"):
        crop = "Expert ROI"
        input_key = remainder[len("expert_roi_") :]
    else:
        raise ValueError(f"Cannot parse crop from run name: {name}")

    if model_key == "jolia_cross_modality":
        feature_set = (
            "Cross-modality image + clinical"
            if has_clinical
            else "Cross-modality image-only"
        )
        benchmark_group = "Cross-modality stress test"
    elif has_clinical:
        feature_set = "Image + clinical"
        benchmark_group = "Image + clinical"
    elif "fusion" in input_key:
        feature_set = "Image-only fused embeddings"
        benchmark_group = "Image-only"
    else:
        feature_set = "Image-only"
        benchmark_group = "Image-only"

    input_label = INPUT_LABELS.get(input_key, input_key)
    if input_key == "selected_fusion" and crop == "Whole volume":
        input_label = "Selected phase fusion"

    return {
        "model_key": model_key,
        "model": MODEL_DISPLAY[model_key],
        "input_key": input_key,
        "input": input_label,
        "crop": crop,
        "feature_set": feature_set,
        "clinical_data": "Yes" if has_clinical else "No",
        "benchmark_group": benchmark_group,
    }


def add_parsed_columns(summary: pd.DataFrame) -> pd.DataFrame:
    parsed = pd.DataFrame.from_records(
        [parse_run_name(run_dir) for run_dir in summary["run_dir"]]
    )
    comparison = pd.concat([parsed, summary.reset_index(drop=True)], axis=1)
    comparison["model_order"] = comparison["model_key"].map(MODEL_ORDER)
    comparison["run_name"] = comparison["run_dir"].map(lambda value: Path(value).name)
    return comparison


def ordered_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    group_order = {
        "Clinical-only": 0,
        "Image-only": 1,
        "Image + clinical": 2,
        "Cross-modality stress test": 3,
    }
    crop_order = {
        "N/A": 0,
        "Whole volume": 1,
        "Expert ROI": 2,
    }
    comparison = comparison.copy()
    comparison["group_order"] = comparison["benchmark_group"].map(group_order)
    comparison["crop_order"] = comparison["crop"].map(crop_order)
    return comparison.sort_values(
        ["model_order", "group_order", "crop_order", "input_key", "run_name"]
    )


def compact_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model",
        "benchmark_group",
        "crop",
        "input",
        "feature_set",
        "clinical_data",
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
        "test_recall_sensitivity",
        "test_specificity",
        "test_precision",
        "test_f1",
        "test_tp",
        "test_fp",
        "test_fn",
        "test_tn",
        "selected_threshold_from_validation",
        "best_c",
        "num_embedding_features",
        "official_train_final_fit",
        "official_test",
        "run_name",
        "run_dir",
        "embeddings",
    ]
    return comparison[columns]


def best_rows_by_metric(comparison: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    scoped = comparison[comparison["benchmark_group"].isin(["Image-only", "Image + clinical"])]
    for model_key, model_df in scoped.groupby("model_key"):
        for group, group_df in model_df.groupby("benchmark_group"):
            for metric_label, metric_col in METRICS.items():
                best = group_df.loc[group_df[metric_col].idxmax()]
                rows.append(
                    {
                        "scope": "per_model",
                        "model": MODEL_DISPLAY[model_key],
                        "benchmark_group": group,
                        "metric": metric_label,
                        "best_value": best[metric_col],
                        "crop": best["crop"],
                        "input": best["input"],
                        "feature_set": best["feature_set"],
                        "clinical_data": best["clinical_data"],
                        "test_roc_auc": best["test_roc_auc"],
                        "test_average_precision": best["test_average_precision"],
                        "test_balanced_accuracy": best["test_balanced_accuracy"],
                        "run_name": best["run_name"],
                    }
                )

    for group, group_df in scoped.groupby("benchmark_group"):
        for metric_label, metric_col in METRICS.items():
            best = group_df.loc[group_df[metric_col].idxmax()]
            rows.append(
                {
                    "scope": "overall",
                    "model": best["model"],
                    "benchmark_group": group,
                    "metric": metric_label,
                    "best_value": best[metric_col],
                    "crop": best["crop"],
                    "input": best["input"],
                    "feature_set": best["feature_set"],
                    "clinical_data": best["clinical_data"],
                    "test_roc_auc": best["test_roc_auc"],
                    "test_average_precision": best["test_average_precision"],
                    "test_balanced_accuracy": best["test_balanced_accuracy"],
                    "run_name": best["run_name"],
                }
            )
    return pd.DataFrame.from_records(rows)


def primary_best_table(
    comparison: pd.DataFrame, benchmark_group: str, metric_col: str = "test_roc_auc"
) -> pd.DataFrame:
    rows = []
    scoped = comparison[comparison["benchmark_group"] == benchmark_group]
    for model_key in (
        "pillar0_breastmri",
        "radiodino",
        "biomedclip",
        "curia",
        "medsiglip",
        "radimagenet",
    ):
        model_df = scoped[scoped["model_key"] == model_key]
        if model_df.empty:
            continue
        best = model_df.loc[model_df[metric_col].idxmax()]
        rows.append(best)
    return pd.DataFrame(rows)


def format_metric(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.3f}"


def markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
    labels: dict[str, str] | None = None,
    metric_cols: set[str] | None = None,
) -> str:
    labels = labels or {}
    metric_cols = metric_cols or set()
    header = [labels.get(col, col) for col in columns]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---:" if col in metric_cols else "---" for col in columns) + "|")
    for _, row in frame.iterrows():
        values = []
        for col in columns:
            if col in metric_cols:
                values.append(format_metric(row[col]))
            else:
                values.append(str(row[col]))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_markdown(
    comparison: pd.DataFrame,
    best_by_metric: pd.DataFrame,
    output_dir: Path,
) -> str:
    clinical = comparison[comparison["model_key"] == "clinical"].iloc[0]
    best_image = primary_best_table(comparison, "Image-only")
    best_multimodal = primary_best_table(comparison, "Image + clinical")

    overall_best = best_by_metric[best_by_metric["scope"] == "overall"].copy()
    overall_best["criterion"] = (
        overall_best["benchmark_group"] + " best " + overall_best["metric"]
    )

    metric_cols = {
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
        "best_value",
    }
    metric_labels = {
        "test_roc_auc": "AUROC",
        "test_average_precision": "AP",
        "test_balanced_accuracy": "Bal Acc",
        "best_value": "Best Value",
    }

    best_image_by_ap = comparison[comparison["benchmark_group"] == "Image-only"].loc[
        comparison[comparison["benchmark_group"] == "Image-only"][
            "test_average_precision"
        ].idxmax()
    ]
    best_image_by_auroc = comparison[
        comparison["benchmark_group"] == "Image-only"
    ].loc[
        comparison[comparison["benchmark_group"] == "Image-only"][
            "test_roc_auc"
        ].idxmax()
    ]
    best_image_by_bal = comparison[
        comparison["benchmark_group"] == "Image-only"
    ].loc[
        comparison[comparison["benchmark_group"] == "Image-only"][
            "test_balanced_accuracy"
        ].idxmax()
    ]
    best_multimodal_by_auroc = comparison[
        comparison["benchmark_group"] == "Image + clinical"
    ].loc[
        comparison[comparison["benchmark_group"] == "Image + clinical"][
            "test_roc_auc"
        ].idxmax()
    ]
    best_multimodal_by_ap = comparison[
        comparison["benchmark_group"] == "Image + clinical"
    ].loc[
        comparison[comparison["benchmark_group"] == "Image + clinical"][
            "test_average_precision"
        ].idxmax()
    ]
    best_multimodal_by_bal = comparison[
        comparison["benchmark_group"] == "Image + clinical"
    ].loc[
        comparison[comparison["benchmark_group"] == "Image + clinical"][
            "test_balanced_accuracy"
        ].idxmax()
    ]

    def display_path(path: Path) -> str:
        try:
            return str(path.relative_to(ROOT.parent))
        except ValueError:
            return str(path)

    def describe(row: pd.Series) -> str:
        if row["crop"] == "N/A":
            return str(row["input"])
        crop = {
            "Whole volume": "whole-volume",
            "Expert ROI": "expert-ROI",
        }.get(str(row["crop"]), str(row["crop"]).lower())
        input_label = {
            "Selected ROI fusion": "selected ROI fusion",
            "Phase 2 - phase 0": "phase2-minus-phase0",
            "Phase 1": "phase1",
        }.get(str(row["input"]), str(row["input"]).lower())
        return f"{row['model']} {crop} {input_label}"

    lines = [
        "# Cross-Model First-Pass Benchmark Summary",
        "",
        f"Snapshot date: {date.today().isoformat()}",
        "",
        "This compares the completed first-pass foundation-model benchmarks using",
        "the same MAMA-MIA official train/test split, frozen encoders, patient-level",
        "embedding aggregation, and L2 logistic regression probe protocol.",
        "",
        "Full tables:",
        "",
        "```text",
        display_path(output_dir / "cross_model_first_pass_comparison.csv"),
        display_path(output_dir / "cross_model_all_results.md"),
        display_path(output_dir / "cross_model_best_by_metric.csv"),
        display_path(output_dir / "cross_model_best_image_only.png"),
        display_path(output_dir / "cross_model_best_image_plus_clinical.png"),
        "```",
        "",
        "## Clinical Baseline",
        "",
        markdown_table(
            pd.DataFrame([clinical]),
            ["model", "test_roc_auc", "test_average_precision", "test_balanced_accuracy"],
            labels=metric_labels,
            metric_cols=metric_cols,
        ),
        "",
        "## Best Image-Only Run Per Model",
        "",
        markdown_table(
            best_image,
            [
                "model",
                "crop",
                "input",
                "test_roc_auc",
                "test_average_precision",
                "test_balanced_accuracy",
            ],
            labels=metric_labels,
            metric_cols=metric_cols,
        ),
        "",
        "## Best Image-Plus-Clinical Run Per Model",
        "",
        markdown_table(
            best_multimodal,
            [
                "model",
                "crop",
                "input",
                "test_roc_auc",
                "test_average_precision",
                "test_balanced_accuracy",
            ],
            labels=metric_labels,
            metric_cols=metric_cols,
        ),
        "",
        "## Overall Metric Winners",
        "",
        markdown_table(
            overall_best,
            [
                "criterion",
                "model",
                "crop",
                "input",
                "best_value",
                "test_roc_auc",
                "test_average_precision",
                "test_balanced_accuracy",
            ],
            labels=metric_labels,
            metric_cols=metric_cols,
        ),
        "",
        "## Interpretation",
        "",
        "The best image-only AUROC is "
        f"{format_metric(best_image_by_auroc['test_roc_auc'])} from "
        f"{describe(best_image_by_auroc)}. The best image-only AP is "
        f"{format_metric(best_image_by_ap['test_average_precision'])} from "
        f"{describe(best_image_by_ap)}, and the best image-only balanced "
        f"accuracy is {format_metric(best_image_by_bal['test_balanced_accuracy'])} "
        f"from {describe(best_image_by_bal)}.",
        "",
        "For multimodal prediction, the best AUROC is "
        f"{format_metric(best_multimodal_by_auroc['test_roc_auc'])} from "
        f"{describe(best_multimodal_by_auroc)}. The best AP is "
        f"{format_metric(best_multimodal_by_ap['test_average_precision'])} "
        f"from {describe(best_multimodal_by_ap)}, and the best selected-threshold "
        "balanced accuracy is "
        f"{format_metric(best_multimodal_by_bal['test_balanced_accuracy'])} "
        f"from {describe(best_multimodal_by_bal)}.",
        "",
        "Clinical-only remains a very strong baseline. The best image-plus-clinical",
        "runs improve AP and/or balanced accuracy, but none of them clearly",
        "dominates clinical-only on all three metrics. That is the key scientific",
        "message for this first-pass benchmark.",
        "",
        "## Recommended Shortlist",
        "",
        "Keep these configurations for deeper validation and later repeated-seed or",
        "cross-validation checks:",
        "",
        f"- {describe(best_multimodal_by_auroc)}: best multimodal AUROC.",
        f"- {describe(best_multimodal_by_ap)}: best multimodal AP.",
        f"- {describe(best_multimodal_by_bal)}: best multimodal balanced accuracy.",
        f"- {describe(best_image_by_auroc)}: best image-only AUROC.",
        "- Clinical-only: mandatory reference baseline.",
    ]
    return "\n".join(lines) + "\n"


def make_all_results_markdown(comparison: pd.DataFrame) -> str:
    metric_cols = {
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
    }
    metric_labels = {
        "benchmark_group": "Group",
        "clinical_data": "Clinical",
        "test_roc_auc": "AUROC",
        "test_average_precision": "AP",
        "test_balanced_accuracy": "Bal Acc",
        "run_name": "Run",
    }
    columns = [
        "model",
        "crop",
        "input",
        "feature_set",
        "clinical_data",
        "test_roc_auc",
        "test_average_precision",
        "test_balanced_accuracy",
        "official_test",
        "run_name",
    ]
    lines = [
        "# Cross-Model All Probe Results",
        "",
        f"Snapshot date: {date.today().isoformat()}",
        "",
        "This table intentionally includes every completed probe run, not only the",
        "headline standardized subset. Use it as the appendix/source table for the",
        "foundation-model benchmark.",
    ]
    for group in (
        "Clinical-only",
        "Image-only",
        "Image + clinical",
        "Cross-modality stress test",
    ):
        group_frame = comparison[comparison["benchmark_group"] == group]
        if group_frame.empty:
            continue
        lines.extend(
            [
                "",
                f"## {group}",
                "",
                markdown_table(
                    group_frame,
                    columns,
                    labels=metric_labels,
                    metric_cols=metric_cols,
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def make_plots(best_image: pd.DataFrame, best_multimodal: pd.DataFrame, output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "matplotlib_cache"))
    import matplotlib.pyplot as plt
    import numpy as np

    plot_specs = [
        ("cross_model_best_image_only.png", "Best Image-Only Run Per Model", best_image),
        (
            "cross_model_best_image_plus_clinical.png",
            "Best Image-Plus-Clinical Run Per Model",
            best_multimodal,
        ),
    ]
    for filename, title, frame in plot_specs:
        labels = frame["model"].tolist()
        x = np.arange(len(labels))
        width = 0.24
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(x - width, frame["test_roc_auc"], width, label="AUROC", color="#2f6f9f")
        ax.bar(x, frame["test_average_precision"], width, label="AP", color="#4f9d69")
        ax.bar(x + width, frame["test_balanced_accuracy"], width, label="Bal Acc", color="#c77d2c")
        ax.set_ylim(0.25, 0.80)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc="upper left", ncols=3, frameon=False)
        ax.grid(axis="y", alpha=0.25)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(args.summary_csv)
    comparison = ordered_comparison(add_parsed_columns(summary))
    compact = compact_columns(comparison)
    best_by_metric = best_rows_by_metric(comparison)
    best_image = primary_best_table(comparison, "Image-only")
    best_multimodal = primary_best_table(comparison, "Image + clinical")

    comparison_path = args.output_dir / "cross_model_first_pass_comparison.csv"
    all_results_path = args.output_dir / "cross_model_all_results.md"
    best_path = args.output_dir / "cross_model_best_by_metric.csv"
    summary_path = args.output_dir / "cross_model_first_pass_summary.md"

    compact.to_csv(comparison_path, index=False)
    all_results_path.write_text(make_all_results_markdown(compact))
    best_by_metric.to_csv(best_path, index=False)
    summary_path.write_text(make_markdown(comparison, best_by_metric, args.output_dir))

    try:
        make_plots(best_image, best_multimodal, args.output_dir)
    except Exception as exc:  # pragma: no cover - plot generation is optional.
        print(f"Warning: could not generate plots: {exc}")

    print(f"Wrote comparison CSV: {comparison_path}")
    print(f"Wrote all-results Markdown: {all_results_path}")
    print(f"Wrote best-by-metric CSV: {best_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
