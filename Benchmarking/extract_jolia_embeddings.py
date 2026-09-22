#!/usr/bin/env python3
"""Extract Jolia global embeddings from breast MRI as a transfer stress test.

Jolia was trained on 3D chest and abdominal CT. This script does not claim
native MRI compatibility: it applies a transparent MRI-to-input adapter and
records that adapter in every output row and summary.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "outputs" / "manifests" / "mamamia_multiphase_foundation_manifest.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "embeddings"
DEFAULT_CACHE_DIR = ROOT / "outputs" / "model_cache"
MODEL_KEY = "jolia_cross_modality"
MODEL_ID = "raidium/Jolia"
MODEL_REVISION = "261abf87b0b1d77e74a75ecf0d6e0ca04043fb01"
TARGET_SIZE = 192
INPUT_CHANNELS = 11
EXPECTED_EMBEDDING_DIM = 576
ADAPTER_NAME = "mri_robust_minmax_replicated_11ch_fit192"

PHASE_ROLES = {
    "phase0": "phase0_path",
    "phase1": "phase1_path",
    "phase2": "phase2_path",
    "last_phase": "last_phase_path",
}
SUBTRACTION_ROLES = {
    "phase1_minus_phase0": ("phase1_path", "phase0_path"),
    "phase2_minus_phase0": ("phase2_path", "phase0_path"),
    "last_phase_minus_phase0": ("last_phase_path", "phase0_path"),
}
INPUT_ROLES = {**PHASE_ROLES, **SUBTRACTION_ROLES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 576-D Jolia global embeddings from 3D breast MRI using "
            "an explicitly out-of-modality 11-channel adapter."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-role", choices=sorted(INPUT_ROLES), default="phase1")
    parser.add_argument("--crop-mode", choices=("whole", "expert_roi"), default="whole")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--usable-flag", default=None)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        choices=("stratified", "head"),
        default="stratified",
        help="How to choose rows when --max-patients is set.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Write partial embeddings every N attempted patients; <=0 disables checkpoints.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patient IDs already present in patient_embeddings.csv.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def set_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((cache_dir / "hub").resolve()))


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def default_usable_flag(crop_mode: str) -> str:
    if crop_mode == "expert_roi":
        return "usable_for_expert_roi_benchmark"
    return "usable_for_multiphase_or_subtraction_benchmark"


def load_manifest(path: Path, usable_flag: str | None, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(path).copy()
    manifest["patient_id"] = manifest["patient_id"].astype(str)
    manifest["pcr"] = pd.to_numeric(manifest["pcr"], errors="coerce")
    eligible = manifest["pcr"].notna() & manifest["split"].isin(["train", "test"])
    if usable_flag:
        if usable_flag not in manifest.columns:
            raise ValueError(f"Usable flag {usable_flag!r} not found in manifest")
        eligible &= manifest[usable_flag].astype(bool)
    if split != "all":
        eligible &= manifest["split"].eq(split)
    rows = manifest[eligible].copy()
    if rows.empty:
        raise ValueError("No eligible rows found for Jolia extraction")
    return rows


def limit_rows(
    rows: pd.DataFrame,
    max_patients: int | None,
    sample_mode: str,
    seed: int,
) -> pd.DataFrame:
    if max_patients is None or max_patients >= len(rows):
        return rows.copy()
    if max_patients <= 0:
        raise ValueError("max_patients must be positive")
    if sample_mode == "head":
        return rows.head(max_patients).copy()

    strata = rows.groupby(["split", "pcr"], dropna=False, group_keys=False)
    pieces = []
    per_stratum = max(1, int(np.ceil(max_patients / max(1, strata.ngroups))))
    for _, group in strata:
        pieces.append(group.sample(n=min(per_stratum, len(group)), random_state=seed))
    sampled = pd.concat(pieces).sample(frac=1.0, random_state=seed)
    if len(sampled) < max_patients:
        remaining = rows.drop(index=sampled.index)
        if not remaining.empty:
            sampled = pd.concat(
                [
                    sampled,
                    remaining.sample(
                        n=min(max_patients - len(sampled), len(remaining)),
                        random_state=seed,
                    ),
                ]
            )
    return sampled.head(max_patients).sort_values(["split", "patient_id"]).copy()


def load_jolia(device: torch.device) -> torch.nn.Module:
    from transformers import AutoModel

    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
        "dtype": torch.float32,
        "revision": MODEL_REVISION,
    }
    try:
        model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
    return model.eval().to(device=device, dtype=torch.float32)


def path_value(row: pd.Series, column: str) -> Path:
    value = row.get(column)
    if pd.isna(value) or not str(value):
        raise ValueError(f"Missing path column {column}")
    path = Path(str(value))
    if not path.exists():
        raise FileNotFoundError(f"Missing image path: {path}")
    return path


def crop_slices(row: pd.Series, crop_mode: str) -> tuple[slice, slice, slice]:
    if crop_mode == "whole":
        return slice(None), slice(None), slice(None)

    starts = [
        int(row["expert_roi_crop_clipped_start_x"]),
        int(row["expert_roi_crop_clipped_start_y"]),
        int(row["expert_roi_crop_clipped_start_z"]),
    ]
    ends = [
        int(row["expert_roi_crop_clipped_end_x"]),
        int(row["expert_roi_crop_clipped_end_y"]),
        int(row["expert_roi_crop_clipped_end_z"]),
    ]
    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        raise ValueError("Invalid expert ROI crop coordinates")
    return slice(starts[0], ends[0]), slice(starts[1], ends[1]), slice(starts[2], ends[2])


def load_array_xyz(row: pd.Series, column: str, crop_mode: str) -> np.ndarray:
    image = nib.load(str(path_value(row, column)))
    array = np.asanyarray(image.dataobj[crop_slices(row, crop_mode)]).astype(np.float32)
    if array.ndim != 3 or min(array.shape) <= 0:
        raise ValueError(f"Expected a non-empty 3D volume, got shape {array.shape}")
    return array


def load_volume(row: pd.Series, input_role: str, crop_mode: str) -> np.ndarray:
    if input_role in PHASE_ROLES:
        return load_array_xyz(row, PHASE_ROLES[input_role], crop_mode)
    if input_role in SUBTRACTION_ROLES:
        post_column, base_column = SUBTRACTION_ROLES[input_role]
        post = load_array_xyz(row, post_column, crop_mode)
        base = load_array_xyz(row, base_column, crop_mode)
        if post.shape != base.shape:
            raise ValueError(f"Subtraction shape mismatch: {post.shape} vs {base.shape}")
        return post - base
    raise ValueError(f"Unsupported input role: {input_role}")


def robust_minmax(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    values = np.clip(values, lo, hi)
    values = (values - lo) / (hi - lo)
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32)


def fit_to_cube(tensor: torch.Tensor, target_size: int = TARGET_SIZE) -> torch.Tensor:
    """Resize a 1x1xDxHxW tensor isotropically, then center-pad to a cube."""
    if tensor.ndim != 5 or tensor.shape[:2] != (1, 1):
        raise ValueError(f"Expected shape (1, 1, D, H, W), got {tuple(tensor.shape)}")
    spatial = tuple(int(value) for value in tensor.shape[-3:])
    scale = float(target_size) / float(max(spatial))
    resized = tuple(max(1, min(target_size, int(round(value * scale)))) for value in spatial)
    tensor = F.interpolate(tensor, size=resized, mode="trilinear", align_corners=False)

    pads: list[int] = []
    for size in reversed(resized):
        total = target_size - size
        before = total // 2
        after = total - before
        pads.extend([before, after])
    tensor = F.pad(tensor, tuple(pads), mode="constant", value=0.0)
    if tensor.shape[-3:] != (target_size, target_size, target_size):
        raise RuntimeError(f"MRI adapter produced unexpected shape {tuple(tensor.shape)}")
    return tensor


def adapt_mri_to_jolia(volume_xyz: np.ndarray) -> torch.Tensor:
    normalized_xyz = robust_minmax(volume_xyz)
    volume_dhw = np.transpose(normalized_xyz, (2, 1, 0)).copy()
    single_channel = torch.from_numpy(volume_dhw).unsqueeze(0).unsqueeze(0).float()
    single_channel = fit_to_cube(single_channel, target_size=TARGET_SIZE)
    return single_channel.repeat(1, INPUT_CHANNELS, 1, 1, 1)


def extract_embedding(
    model: torch.nn.Module,
    adapted_volume: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    adapted_volume = adapted_volume.to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        output = model(adapted_volume)
    embedding = getattr(output, "pooler_output", None)
    if embedding is None and isinstance(output, dict):
        embedding = output.get("pooler_output")
    if embedding is None:
        raise TypeError("Jolia output does not contain pooler_output")
    if embedding.ndim != 2 or embedding.shape[0] != 1:
        raise ValueError(f"Unexpected Jolia pooler_output shape: {tuple(embedding.shape)}")
    embedding = embedding.squeeze(0).detach().cpu().float().numpy()
    if embedding.size != EXPECTED_EMBEDDING_DIM:
        raise ValueError(
            f"Expected {EXPECTED_EMBEDDING_DIM} Jolia features, got {embedding.size}"
        )
    return embedding


def feature_record(embedding: np.ndarray) -> dict[str, float]:
    return {f"emb_{index:04d}": float(value) for index, value in enumerate(embedding)}


def output_paths(output_root: Path, crop_mode: str, input_role: str) -> tuple[Path, Path]:
    output_dir = output_root / MODEL_KEY / f"{crop_mode}_{input_role}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "patient_embeddings.csv", output_dir / "summary.json"


def load_existing_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    table = pd.read_csv(path)
    if "adapter" in table and not table["adapter"].eq(ADAPTER_NAME).all():
        raise ValueError(f"Existing output at {path} uses a different MRI adapter")
    return [dict(record) for record in table.to_dict(orient="records")]


def write_embeddings(path: Path, records: list[dict[str, object]]) -> pd.DataFrame:
    embeddings = pd.DataFrame.from_records(records)
    embeddings.to_csv(path, index=False)
    return embeddings


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def extract_patient_embedding(
    row: pd.Series,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    volume = load_volume(row, args.input_role, args.crop_mode)
    source_shape = tuple(int(value) for value in volume.shape)
    adapted = adapt_mri_to_jolia(volume)
    embedding = extract_embedding(model, adapted, device)
    record: dict[str, object] = {
        "patient_id": str(row["patient_id"]),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "pcr": int(row["pcr"]),
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "benchmark_scope": "cross_modality_stress_test",
        "source_modality": "breast_dce_mri",
        "pretraining_modality": "adult_chest_abdominal_ct",
        "adapter": ADAPTER_NAME,
        "source_shape_x": source_shape[0],
        "source_shape_y": source_shape[1],
        "source_shape_z": source_shape[2],
        "input_shape": f"1x{INPUT_CHANNELS}x{TARGET_SIZE}x{TARGET_SIZE}x{TARGET_SIZE}",
    }
    record.update(feature_record(embedding))
    return record


def main() -> None:
    args = parse_args()
    args.cache_dir = args.cache_dir.resolve()
    set_hf_cache(args.cache_dir)

    device = choose_device(args.device)
    usable_flag = args.usable_flag or default_usable_flag(args.crop_mode)
    rows = load_manifest(args.manifest, usable_flag=usable_flag, split=args.split)
    rows = rows.sort_values(["split", "patient_id"]).reset_index(drop=True)
    rows = limit_rows(rows, args.max_patients, args.sample_mode, args.seed).reset_index(drop=True)

    embeddings_path, summary_path = output_paths(
        args.output_root, args.crop_mode, args.input_role
    )
    records: list[dict[str, object]] = []
    completed_patient_ids: set[str] = set()
    if args.resume:
        records = load_existing_records(embeddings_path)
        completed_patient_ids = {
            str(record["patient_id"])
            for record in records
            if "patient_id" in record and not pd.isna(record["patient_id"])
        }
        if completed_patient_ids:
            print(
                f"Resuming from {embeddings_path} with "
                f"{len(completed_patient_ids)} existing embeddings.",
                flush=True,
            )

    print(
        "Jolia MRI transfer stress test: the model was trained on CT; "
        f"adapter={ADAPTER_NAME}; device={device}.",
        flush=True,
    )
    model = load_jolia(device)

    failures: list[dict[str, object]] = []
    log_every = max(1, int(args.log_every))
    checkpoint_every = int(args.checkpoint_every)
    rows_skipped_resume = 0
    rows_newly_embedded = 0
    for index, row in rows.iterrows():
        patient_id = str(row["patient_id"])
        if patient_id in completed_patient_ids:
            rows_skipped_resume += 1
            if (index + 1) % log_every == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] skipped existing {patient_id}", flush=True)
            continue
        try:
            records.append(extract_patient_embedding(row, model, args, device))
            completed_patient_ids.add(patient_id)
            rows_newly_embedded += 1
            if (index + 1) % log_every == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] embedded {patient_id}", flush=True)
            if checkpoint_every > 0 and (index + 1) % checkpoint_every == 0:
                write_embeddings(embeddings_path, records)
                print(
                    f"[{index + 1}/{len(rows)}] checkpointed {len(records)} embeddings "
                    f"to {embeddings_path}",
                    flush=True,
                )
        except Exception as exc:  # noqa: BLE001 - preserve progress across patient failures.
            failures.append({"patient_id": patient_id, "error": str(exc)})
            print(f"[{index + 1}/{len(rows)}] failed {patient_id}: {exc}", flush=True)
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if not records:
        raise RuntimeError(f"No Jolia embeddings were extracted. Failures: {failures[:5]}")

    embeddings = write_embeddings(embeddings_path, records)
    summary = {
        "manifest": str(args.manifest),
        "embeddings": str(embeddings_path),
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "usable_flag": usable_flag,
        "split": args.split,
        "max_patients": args.max_patients,
        "sample_mode": args.sample_mode,
        "device": str(device),
        "torch_dtype": "float32",
        "checkpoint_every": checkpoint_every,
        "rows_requested": int(len(rows)),
        "rows_embedded": int(len(records)),
        "rows_newly_embedded": int(rows_newly_embedded),
        "rows_skipped_resume": int(rows_skipped_resume),
        "rows_failed": int(len(failures)),
        "failures": failures,
        "embedding_dim": int(sum(column.startswith("emb_") for column in embeddings.columns)),
        "benchmark_scope": "cross_modality_stress_test",
        "adapter": ADAPTER_NAME,
        "adapter_description": (
            "Robust 0.5/99.5 percentile MRI scaling to [0,1], aspect-preserving "
            "fit with center padding to 192 cubed, then replication into Jolia's "
            "11 fixed CT-window channels."
        ),
        "model_card_limitation": (
            "Jolia was trained on adult chest/abdominal CT and is not expected "
            "to generalize to other modalities."
        ),
        "leaderboard_policy": "report_separately_from_modality_matched_primary_models",
    }
    write_json(summary_path, summary)
    print(f"Wrote embeddings: {embeddings_path}")
    print(f"Wrote summary:    {summary_path}")


if __name__ == "__main__":
    main()
