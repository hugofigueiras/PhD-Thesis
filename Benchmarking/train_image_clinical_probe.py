#!/usr/bin/env python3
"""Train a logistic pCR probe on image embeddings plus clinical variables."""

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
    Path(__file__).resolve().parent / "outputs" / "probes" / "image_clinical_logreg"
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
METADATA_COLUMNS = {
    "patient_id",
    "case_id",
    "dataset",
    "split",
    "pcr",
    "model_key",
    "input_role",
    "crop_mode",
    "aggregation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regularized logistic pCR probe on image embeddings plus clinical variables."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--usable-flag", default="usable_for_whole_volume_benchmark")
    parser.add_argument("--feature-prefix", default="emb_")
    parser.add_argument(
        "--clinical-columns",
        nargs="+",
        default=DEFAULT_CLINICAL_COLUMNS,
        help="Baseline clinical columns to use. Missing columns are ignored.",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c-grid", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("average_precision", "roc_auc", "balanced_accuracy"),
        default="average_precision",
    )
    return parser.parse_args()


def load_embeddings(path: Path, feature_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    embeddings = pd.read_csv(path)
    if "patient_id" not in embeddings.columns:
        raise ValueError(f"{path} must contain patient_id")
    embeddings = embeddings.copy()
    embeddings["patient_id"] = embeddings["patient_id"].astype(str)
    if embeddings["patient_id"].duplicated().any():
        duplicated = embeddings.loc[embeddings["patient_id"].duplicated(), "patient_id"].head()
        raise ValueError(f"Embeddings must have one row per patient. Duplicates: {duplicated.tolist()}")

    if feature_prefix:
        feature_columns = [c for c in embeddings.columns if c.startswith(feature_prefix)]
    else:
        feature_columns = [
            c
            for c in embeddings.columns
            if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(embeddings[c])
        ]
    if not feature_columns:
        raise ValueError("No numeric embedding feature columns found")
    return embeddings, feature_columns


def usable_clinical_columns(rows: pd.DataFrame, requested_columns: list[str]) -> list[str]:
    columns = [column for column in requested_columns if column in rows.columns]
    columns = [column for column in columns if rows[column].notna().any()]
    if not columns:
        raise ValueError("No requested clinical columns are present and non-empty")
    return columns


def split_feature_types(rows: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    for column in columns:
        if pd.api.types.is_numeric_dtype(rows[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)
    manifest = load_manifest(args.manifest, usable_flag=args.usable_flag)
    embeddings, embedding_columns = load_embeddings(args.embeddings, args.feature_prefix)
    rows = manifest.merge(
        embeddings[["patient_id", *embedding_columns]],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if rows.empty:
        raise ValueError("No manifest rows matched the embedding patient_id values")
    if rows[embedding_columns].isna().any().any():
        missing = rows[embedding_columns].isna().sum()
        missing = missing[missing > 0].to_dict()
        raise ValueError(f"Embedding features contain missing values: {missing}")

    train_index, val_index, test_index = split_official_train_validation(
        rows=rows,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_rows = rows.loc[train_index].copy()
    val_rows = rows.loc[val_index].copy()
    test_rows = rows.loc[test_index].copy()
    final_train_rows = rows[rows["split"] == "train"].copy()

    clinical_columns = usable_clinical_columns(train_rows, list(args.clinical_columns))
    numeric_clinical_columns, categorical_clinical_columns = split_feature_types(
        train_rows, clinical_columns
    )
    numeric_columns = [*embedding_columns, *numeric_clinical_columns]
    categorical_columns = categorical_clinical_columns
    feature_columns = [*embedding_columns, *clinical_columns]
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
        "probe": "image_clinical_logistic_regression",
        "manifest": str(args.manifest),
        "embeddings": str(args.embeddings),
        "usable_flag": args.usable_flag,
        "selection_metric": args.selection_metric,
        "best_c": best_c,
        "selected_threshold_from_validation": selected_threshold,
        "num_embedding_features": int(len(embedding_columns)),
        "num_clinical_features_requested": int(len(args.clinical_columns)),
        "clinical_columns": clinical_columns,
        "numeric_clinical_columns": numeric_clinical_columns,
        "categorical_clinical_columns": categorical_clinical_columns,
        "split_counts": {
            "matched_rows": int(len(rows)),
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
            "embeddings": str(args.embeddings),
            "output_dir": str(output_dir),
            "usable_flag": args.usable_flag,
            "feature_prefix": args.feature_prefix,
            "clinical_columns_requested": list(args.clinical_columns),
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "c_grid": c_values,
            "selection_metric": args.selection_metric,
        },
    )

    print(f"Wrote metrics:     {output_dir / 'metrics.json'}")
    print(f"Wrote predictions: {output_dir / 'test_predictions.csv'}")
    print(f"Wrote model:       {output_dir / 'model.joblib'}")
    print(
        "Official test AUROC/AP/balanced accuracy: "
        f"{metrics['official_test_metrics_selected_threshold']['roc_auc']:.4f} / "
        f"{metrics['official_test_metrics_selected_threshold']['average_precision']:.4f} / "
        f"{metrics['official_test_metrics_selected_threshold']['balanced_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()
