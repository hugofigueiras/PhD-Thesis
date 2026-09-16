"""Shared utilities for lightweight pCR benchmark probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


DEFAULT_C_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_c_grid(value: str | None) -> list[float]:
    if not value:
        return list(DEFAULT_C_GRID)
    c_values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not c_values:
        raise ValueError("C grid cannot be empty")
    return c_values


def load_manifest(path: Path, usable_flag: str | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"patient_id", "split", "pcr"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    manifest = manifest.copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["pcr"] = pd.to_numeric(manifest["pcr"], errors="coerce")
    manifest["split"] = manifest["split"].astype(str)

    eligible = manifest["pcr"].notna() & manifest["split"].isin(["train", "test"])
    if usable_flag:
        if usable_flag not in manifest.columns:
            raise ValueError(f"Usable flag {usable_flag!r} not found in manifest")
        eligible &= manifest[usable_flag].astype(bool)
    return manifest[eligible].copy()


def split_official_train_validation(
    rows: pd.DataFrame,
    validation_fraction: float,
    seed: int,
) -> tuple[pd.Index, pd.Index, pd.Index]:
    train_rows = rows[rows["split"] == "train"]
    test_rows = rows[rows["split"] == "test"]
    if train_rows.empty or test_rows.empty:
        raise ValueError("Manifest must contain non-empty official train and test rows")

    y_train = train_rows["pcr"].astype(int)
    train_index, val_index = train_test_split(
        train_rows.index,
        test_size=validation_fraction,
        random_state=seed,
        stratify=y_train,
    )
    return pd.Index(train_index), pd.Index(val_index), pd.Index(test_rows.index)


def safe_metrics(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    threshold: float,
) -> dict[str, float | int | None]:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_prob_arr = np.asarray(list(y_prob), dtype=float)
    y_pred = (y_prob_arr >= threshold).astype(int)

    unique = np.unique(y_true_arr)
    roc_auc = float(roc_auc_score(y_true_arr, y_prob_arr)) if unique.size == 2 else None
    average_precision = (
        float(average_precision_score(y_true_arr, y_prob_arr)) if unique.size == 2 else None
    )
    try:
        loss = float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1]))
    except ValueError:
        loss = None

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(recall_score(y_true_arr, y_pred, zero_division=0))
    specificity = float(tn / (tn + fp)) if (tn + fp) else None
    return {
        "rows": int(y_true_arr.size),
        "pcr0": int((y_true_arr == 0).sum()),
        "pcr1": int((y_true_arr == 1).sum()),
        "threshold": float(threshold),
        "loss": loss,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall_sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "pred0": int((y_pred == 0).sum()),
        "pred1": int((y_pred == 1).sum()),
    }


def choose_threshold(
    y_true: Iterable[int],
    y_prob: Iterable[float],
    grid_size: int = 501,
) -> tuple[float, dict[str, float | int | None]]:
    thresholds = np.linspace(0.0, 1.0, grid_size)
    best_threshold = 0.5
    best_metrics = safe_metrics(y_true, y_prob, best_threshold)
    best_score = float(best_metrics["balanced_accuracy"])

    for threshold in thresholds:
        current_metrics = safe_metrics(y_true, y_prob, float(threshold))
        current_score = float(current_metrics["balanced_accuracy"])
        if current_score > best_score:
            best_score = current_score
            best_threshold = float(threshold)
            best_metrics = current_metrics
    return best_threshold, best_metrics


def select_best_by_metric(
    candidates: list[dict[str, object]],
    metric: str,
) -> dict[str, object]:
    def score(candidate: dict[str, object]) -> tuple[float, float]:
        metrics = candidate["validation_metrics_0p5"]
        assert isinstance(metrics, dict)
        value = metrics.get(metric)
        fallback = metrics.get("roc_auc")
        primary = -np.inf if value is None else float(value)
        secondary = -np.inf if fallback is None else float(fallback)
        return primary, secondary

    return max(candidates, key=score)


def make_logistic_model(c_value: float) -> LogisticRegression:
    return LogisticRegression(
        C=float(c_value),
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
    )


def make_numeric_embedding_pipeline(c_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", make_logistic_model(c_value)),
        ]
    )


def make_clinical_pipeline(
    numeric_columns: list[str],
    categorical_columns: list[str],
    c_value: float,
) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_columns,
            )
        )
    if not transformers:
        raise ValueError("At least one numeric or categorical column is required")

    return Pipeline(
        steps=[
            ("preprocess", ColumnTransformer(transformers=transformers)),
            ("classifier", make_logistic_model(c_value)),
        ]
    )


def predict_positive_probability(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(features)
    return probabilities[:, 1]


def run_c_grid_search(
    make_pipeline,
    train_features: pd.DataFrame,
    train_labels: pd.Series,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    c_values: list[float],
    selection_metric: str,
) -> tuple[float, float, list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    for c_value in c_values:
        model = make_pipeline(c_value)
        model.fit(train_features, train_labels)
        val_prob = predict_positive_probability(model, validation_features)
        threshold, threshold_metrics = choose_threshold(validation_labels, val_prob)
        candidates.append(
            {
                "c": float(c_value),
                "validation_threshold": float(threshold),
                "validation_metrics_0p5": safe_metrics(validation_labels, val_prob, 0.5),
                "validation_metrics_selected_threshold": threshold_metrics,
            }
        )

    best = select_best_by_metric(candidates, selection_metric)
    return float(best["c"]), float(best["validation_threshold"]), candidates


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_predictions(
    path: Path,
    rows: pd.DataFrame,
    y_prob: Iterable[float],
    threshold: float,
) -> None:
    output_columns = [c for c in ["patient_id", "dataset", "split", "pcr"] if c in rows.columns]
    predictions = rows[output_columns].copy()
    predictions["pred_prob"] = np.asarray(list(y_prob), dtype=float)
    predictions["pred_label"] = (predictions["pred_prob"] >= threshold).astype(int)
    predictions["threshold"] = float(threshold)
    predictions.to_csv(path, index=False)


def save_model(path: Path, model: Pipeline) -> None:
    joblib.dump(model, path)


def get_feature_names(model: Pipeline, input_columns: list[str]) -> list[str]:
    if "preprocess" not in model.named_steps:
        return input_columns
    preprocess = model.named_steps["preprocess"]
    try:
        return [str(v) for v in preprocess.get_feature_names_out()]
    except Exception:
        return input_columns


def write_coefficients(
    path: Path,
    model: Pipeline,
    input_columns: list[str],
) -> None:
    classifier = model.named_steps["classifier"]
    names = get_feature_names(model, input_columns)
    coefficients = np.ravel(classifier.coef_)
    if len(names) != len(coefficients):
        names = [f"feature_{i}" for i in range(len(coefficients))]
    pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=False).to_csv(path, index=False)
