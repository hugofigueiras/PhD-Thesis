#!/usr/bin/env python3
"""Train a leakage-safe clinical-only logistic probe for pCR prediction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from probe_utils import (
    ensure_output_dir,
    load_manifest,
    make_clinical_pipeline,
    parse_c_grid,
    predict_positive_probability,
    run_c_grid_search,
    safe_metrics,
    save_model,
    split_official_train_validation,
    write_coefficients,
    write_json,
    write_predictions,
)


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "outputs"
    / "manifests"
    / "mamamia_multiphase_foundation_manifest.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "outputs" / "probes" / "clinical_logreg"
)

DEFAULT_CLINICAL_COLUMNS = [
    "age",
    "menopause",
    "er",
    "pr",
    "hr",
    "her2",
    "tumor_subtype",
    "nottingham_grade",
    "mammaprint",
    "oncotype_score",
    "bilateral_breast_cancer",
    "multifocal_cancer",
    "ethnicity",
    "has_implant",
    "bmi_group",
    "breast_density",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regularized logistic-regression clinical pCR probe."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--usable-flag",
        default="usable_for_whole_volume_benchmark",
        help="Manifest boolean column used to select eligible labeled patients.",
    )
    parser.add_argument(
        "--clinical-columns",
        nargs="+",
        default=DEFAULT_CLINICAL_COLUMNS,
        help="Baseline clinical columns to use. Missing columns are ignored.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--c-grid",
        default=None,
        help="Comma-separated logistic-regression C values. Defaults to a small log grid.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("average_precision", "roc_auc", "balanced_accuracy"),
        default="average_precision",
    )
    return parser.parse_args()


def usable_columns(rows: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    present = [column for column in requested_columns if column in rows.columns]
    non_empty = [column for column in present if rows[column].notna().any()]
    if not non_empty:
        raise ValueError("No requested clinical columns are present and non-empty")
    return non_empty


def split_feature_types(rows: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in columns:
        values = rows[column]
        if pd.api.types.is_numeric_dtype(values):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    rows = load_manifest(args.manifest, usable_flag=args.usable_flag)

    train_index, val_index, test_index = split_official_train_validation(
        rows=rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_rows = rows.loc[train_index].copy()
    val_rows = rows.loc[val_index].copy()
    test_rows = rows.loc[test_index].copy()
    final_train_rows = rows[rows["split"] == "train"].copy()

    feature_columns = usable_columns(train_rows, list(args.clinical_columns))
    numeric_columns, categorical_columns = split_feature_types(train_rows, feature_columns)
    c_values = parse_c_grid(args.c_grid)

    def make_pipeline(c_value: float):
        return make_clinical_pipeline(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            c_value=c_value,
        )

    best_c, selected_threshold, candidates = run_c_grid_search(
        make_pipeline=make_pipeline,
        train_features=train_rows[feature_columns],
        train_labels=train_rows["pcr"].astype(int),
        validation_features=val_rows[feature_columns],
        validation_labels=val_rows["pcr"].astype(int),
        c_values=c_values,
        selection_metric=args.selection_metric,
    )

    final_model = make_pipeline(best_c)
    final_model.fit(final_train_rows[feature_columns], final_train_rows["pcr"].astype(int))

    train_prob = predict_positive_probability(final_model, final_train_rows[feature_columns])
    test_prob = predict_positive_probability(final_model, test_rows[feature_columns])

    metrics = {
        "probe": "clinical_logistic_regression",
        "manifest": str(args.manifest),
        "usable_flag": args.usable_flag,
        "selection_metric": args.selection_metric,
        "best_c": best_c,
        "selected_threshold_from_validation": selected_threshold,
        "feature_columns": feature_columns,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "split_counts": {
            "train_sub": int(len(train_rows)),
            "validation": int(len(val_rows)),
            "official_train_final_fit": int(len(final_train_rows)),
            "official_test": int(len(test_rows)),
        },
        "candidates": candidates,
        "official_train_metrics_selected_threshold": safe_metrics(
            final_train_rows["pcr"].astype(int), train_prob, selected_threshold
        ),
        "official_train_metrics_0p5": safe_metrics(
            final_train_rows["pcr"].astype(int), train_prob, 0.5
        ),
        "official_test_metrics_selected_threshold": safe_metrics(
            test_rows["pcr"].astype(int), test_prob, selected_threshold
        ),
        "official_test_metrics_0p5": safe_metrics(test_rows["pcr"].astype(int), test_prob, 0.5),
    }

    write_json(output_dir / "metrics.json", metrics)
    write_predictions(
        output_dir / "test_predictions.csv",
        rows=test_rows,
        y_prob=test_prob,
        threshold=selected_threshold,
    )
    write_predictions(
        output_dir / "train_predictions.csv",
        rows=final_train_rows,
        y_prob=train_prob,
        threshold=selected_threshold,
    )
    write_coefficients(output_dir / "coefficients.csv", final_model, feature_columns)
    save_model(output_dir / "model.joblib", final_model)
    write_json(
        output_dir / "run_config.json",
        {
            "manifest": str(args.manifest),
            "output_dir": str(output_dir),
            "usable_flag": args.usable_flag,
            "clinical_columns_requested": list(args.clinical_columns),
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "c_grid": c_values,
            "selection_metric": args.selection_metric,
        },
    )

    print(f"Wrote metrics:      {output_dir / 'metrics.json'}")
    print(f"Wrote predictions:  {output_dir / 'test_predictions.csv'}")
    print(f"Wrote coefficients: {output_dir / 'coefficients.csv'}")
    print(f"Wrote model:        {output_dir / 'model.joblib'}")
    print(
        "Official test AUROC/AP/balanced accuracy: "
        f"{metrics['official_test_metrics_selected_threshold']['roc_auc']:.4f} / "
        f"{metrics['official_test_metrics_selected_threshold']['average_precision']:.4f} / "
        f"{metrics['official_test_metrics_selected_threshold']['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
