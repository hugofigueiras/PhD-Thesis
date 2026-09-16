#!/usr/bin/env python3
"""Concatenate multiple patient-level embedding CSVs into one fused table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parent
    / "outputs"
    / "embeddings"
    / "fused"
    / "patient_embeddings.csv"
)
METADATA_COLUMNS = [
    "patient_id",
    "dataset",
    "split",
    "pcr",
    "model_key",
    "model_id",
    "input_role",
    "crop_mode",
    "aggregation",
    "num_slices",
    "slice_indices",
    "source_shape_x",
    "source_shape_y",
    "source_shape_z",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fuse multiple patient-level embedding CSVs by patient_id. Inputs "
            "must be provided as label=path pairs."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        metavar="LABEL=CSV",
        help="Embedding input with a short label, e.g. phase0=.../patient_embeddings.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-prefix",
        default="emb_",
        help="Feature-column prefix in input files.",
    )
    return parser.parse_args()


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Input must be LABEL=CSV, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Input label cannot be empty: {value}")
    return label, Path(path.strip())


def load_embedding_table(path: Path, feature_prefix: str) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_csv(path)
    if "patient_id" not in table.columns:
        raise ValueError(f"{path} must contain patient_id")
    table = table.copy()
    table["patient_id"] = table["patient_id"].astype(str)
    if table["patient_id"].duplicated().any():
        duplicated = table.loc[table["patient_id"].duplicated(), "patient_id"].head()
        raise ValueError(f"{path} has duplicate patients: {duplicated.tolist()}")
    feature_columns = [column for column in table.columns if column.startswith(feature_prefix)]
    if not feature_columns:
        raise ValueError(f"{path} has no feature columns starting with {feature_prefix!r}")
    return table, feature_columns


def metadata_frame(table: pd.DataFrame) -> pd.DataFrame:
    columns = [column for column in METADATA_COLUMNS if column in table.columns]
    return table[columns].copy()


def feature_frame(
    table: pd.DataFrame,
    feature_columns: list[str],
    label: str,
) -> pd.DataFrame:
    features = table[["patient_id", *feature_columns]].copy()
    rename = {
        column: f"emb_{label}_{column.removeprefix('emb_')}" for column in feature_columns
    }
    return features.rename(columns=rename)


def main() -> None:
    args = parse_args()
    parsed_inputs = [parse_input(value) for value in args.input]
    if len({label for label, _ in parsed_inputs}) != len(parsed_inputs):
        raise ValueError("Input labels must be unique")

    merged: pd.DataFrame | None = None
    input_summaries: list[dict[str, object]] = []
    for idx, (label, path) in enumerate(parsed_inputs):
        table, feature_columns = load_embedding_table(path, args.feature_prefix)
        if idx == 0:
            merged = metadata_frame(table)
        assert merged is not None
        features = feature_frame(table, feature_columns, label)
        merged = merged.merge(features, on="patient_id", how="inner", validate="one_to_one")
        input_summaries.append(
            {
                "label": label,
                "path": str(path),
                "rows": int(len(table)),
                "feature_columns": int(len(feature_columns)),
            }
        )

    assert merged is not None
    feature_count = int(sum(column.startswith("emb_") for column in merged.columns))
    if feature_count == 0:
        raise RuntimeError("Fused table has no embedding features")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    summary = {
        "output": str(args.output),
        "rows": int(len(merged)),
        "feature_columns": feature_count,
        "inputs": input_summaries,
        "feature_prefix": "emb_",
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote fused embeddings: {args.output}")
    print(f"Wrote summary:          {summary_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
