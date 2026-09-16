#!/usr/bin/env python3
"""Extract patient-level embeddings from the 3D Pillar-0 BreastMRI model."""

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
MODEL_KEY = "pillar0_breastmri"
MODEL_ID = "YalaLab/Pillar0-BreastMRI"
TARGET_HW = 384
TARGET_D = 192

RAW_INPUTS = {
    "phase0_phase1_phase2": ("phase0_path", "phase1_path", "phase2_path"),
    "phase0_phase1_last": ("phase0_path", "phase1_path", "last_phase_path"),
    "phase0_phase2_last": ("phase0_path", "phase2_path", "last_phase_path"),
}
SUBTRACTION_INPUTS = {
    "subtractions_phase1_phase2_last": (
        ("phase1_path", "phase0_path"),
        ("phase2_path", "phase0_path"),
        ("last_phase_path", "phase0_path"),
    ),
}
INPUT_ROLES = {**RAW_INPUTS, **SUBTRACTION_INPUTS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 1152-D patient embeddings from 3-channel 3D DCE-MRI "
            "volumes with Pillar-0 BreastMRI."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--input-role", choices=sorted(INPUT_ROLES), default="phase0_phase1_last")
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
    parser.add_argument("--target-height", type=int, default=TARGET_HW)
    parser.add_argument("--target-width", type=int, default=TARGET_HW)
    parser.add_argument("--target-depth", type=int, default=TARGET_D)
    parser.add_argument(
        "--resample-spacing-mm",
        type=float,
        default=1.0,
        help="Isotropic spacing before center pad/crop. Set <=0 to disable resampling.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help=(
            "Torch dtype for Pillar-0. Auto uses float32 because the current "
            "remote-code model is not half-precision safe."
        ),
    )
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help=(
            "Write partial patient_embeddings.csv progress every N attempted "
            "patients. Set <=0 to write only at the end."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip patient IDs already present in the output patient_embeddings.csv.",
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


def choose_torch_dtype(value: str, device: torch.device) -> torch.dtype:
    if value == "auto":
        return torch.float32
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {value}")


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
        raise ValueError("No eligible rows found for Pillar-0 extraction")
    return rows


def limit_rows(
    rows: pd.DataFrame,
    max_patients: int | None,
    sample_mode: str,
    seed: int,
) -> pd.DataFrame:
    if max_patients is None or max_patients >= len(rows):
        return rows.copy()
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


def load_pillar0(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from transformers import AutoModel, PreTrainedModel

    # Pillar-0's remote-code model is compatible with older Transformers
    # releases. Transformers 5.x expects this property during checkpoint
    # finalization, so provide a narrow compatibility alias before loading.
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        def all_tied_weights_keys(self) -> dict[str, object]:
            keys = getattr(self, "_tied_weights_keys", None)
            if keys is None:
                return {}
            if isinstance(keys, dict):
                return keys
            return {str(key): key for key in keys}

        PreTrainedModel.all_tied_weights_keys = property(all_tied_weights_keys)  # type: ignore[attr-defined]

    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": False,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
    except TypeError:
        if "dtype" not in kwargs:
            raise
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = AutoModel.from_pretrained(MODEL_ID, **kwargs)
    return model.eval().to(device)


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


def load_array_xyz(row: pd.Series, column: str, crop_mode: str) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = nib.load(str(path_value(row, column)))
    crop = crop_slices(row, crop_mode)
    array = np.asanyarray(image.dataobj[crop]).astype(np.float32)
    spacing_xyz = tuple(float(v) for v in image.header.get_zooms()[:3])
    return array, spacing_xyz


def xyz_to_dhw(array: np.ndarray, spacing_xyz: tuple[float, float, float]) -> tuple[np.ndarray, tuple[float, float, float]]:
    dx, dy, dz = spacing_xyz
    return np.transpose(array, (2, 1, 0)), (dz, dy, dx)


def resample_dhw(
    volume: np.ndarray,
    spacing_dhw: tuple[float, float, float],
    target_spacing_mm: float,
) -> np.ndarray:
    import SimpleITK as sitk

    if target_spacing_mm <= 0:
        return volume.astype(np.float32, copy=False)

    depth, height, width = volume.shape
    spacing_d, spacing_h, spacing_w = spacing_dhw
    new_size = [
        max(1, int(round(width * spacing_w / target_spacing_mm))),
        max(1, int(round(height * spacing_h / target_spacing_mm))),
        max(1, int(round(depth * spacing_d / target_spacing_mm))),
    ]
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing([spacing_w, spacing_h, spacing_d])

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing_mm] * 3)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    values = np.asarray(volume, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    lo, hi = np.percentile(finite, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(finite))
        hi = float(np.max(finite))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float32)
    values = np.clip(values, lo, hi)
    values = (values - lo) / (hi - lo + 1e-8)
    values[~np.isfinite(values)] = 0.0
    return values.astype(np.float32)


def center_pad_or_crop_3d(
    tensor: torch.Tensor,
    target_depth: int,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    depth, height, width = tensor.shape
    if depth < target_depth:
        front = (target_depth - depth) // 2
        back = target_depth - depth - front
        tensor = F.pad(tensor, (0, 0, 0, 0, front, back))
    elif depth > target_depth:
        start = (depth - target_depth) // 2
        tensor = tensor[start : start + target_depth]

    depth, height, width = tensor.shape
    if height < target_height:
        top = (target_height - height) // 2
        bottom = target_height - height - top
        tensor = F.pad(tensor, (0, 0, top, bottom, 0, 0))
    elif height > target_height:
        start = (height - target_height) // 2
        tensor = tensor[:, start : start + target_height, :]

    depth, height, width = tensor.shape
    if width < target_width:
        left = (target_width - width) // 2
        right = target_width - width - left
        tensor = F.pad(tensor, (left, right, 0, 0, 0, 0))
    elif width > target_width:
        start = (width - target_width) // 2
        tensor = tensor[:, :, start : start + target_width]
    return tensor


def preprocess_channel(
    array_xyz: np.ndarray,
    spacing_xyz: tuple[float, float, float],
    args: argparse.Namespace,
) -> torch.Tensor:
    volume_dhw, spacing_dhw = xyz_to_dhw(array_xyz, spacing_xyz)
    volume_dhw = resample_dhw(
        volume=volume_dhw,
        spacing_dhw=spacing_dhw,
        target_spacing_mm=float(args.resample_spacing_mm),
    )
    volume_dhw = normalize_volume(volume_dhw)
    tensor_dhw = torch.from_numpy(volume_dhw).float()
    tensor_dhw = center_pad_or_crop_3d(
        tensor=tensor_dhw,
        target_depth=int(args.target_depth),
        target_height=int(args.target_height),
        target_width=int(args.target_width),
    )
    return tensor_dhw.permute(1, 2, 0).contiguous()


def build_input_volume(
    row: pd.Series,
    input_role: str,
    crop_mode: str,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, object]]:
    channels = []
    channel_names = []
    source_shapes = []

    if input_role in RAW_INPUTS:
        for column in RAW_INPUTS[input_role]:
            array_xyz, spacing_xyz = load_array_xyz(row, column, crop_mode)
            channels.append(preprocess_channel(array_xyz, spacing_xyz, args))
            channel_names.append(column.removesuffix("_path"))
            source_shapes.append("x".join(str(int(v)) for v in array_xyz.shape))
    elif input_role in SUBTRACTION_INPUTS:
        for post_column, base_column in SUBTRACTION_INPUTS[input_role]:
            post, spacing_xyz = load_array_xyz(row, post_column, crop_mode)
            base, _ = load_array_xyz(row, base_column, crop_mode)
            if post.shape != base.shape:
                raise ValueError(f"Subtraction shape mismatch: {post.shape} vs {base.shape}")
            channels.append(preprocess_channel(post - base, spacing_xyz, args))
            channel_names.append(f"{post_column.removesuffix('_path')}_minus_{base_column.removesuffix('_path')}")
            source_shapes.append("x".join(str(int(v)) for v in post.shape))
    else:
        raise ValueError(f"Unsupported input role: {input_role}")

    volume = torch.stack(channels, dim=0).unsqueeze(0)
    metadata = {
        "input_channels": ";".join(channel_names),
        "source_shapes_xyz": ";".join(source_shapes),
        "input_shape": "x".join(str(int(v)) for v in volume.shape),
    }
    return volume, metadata


def output_tensor(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if value is not None:
                return value
    for attribute in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value
    if isinstance(output, (tuple, list)):
        return output[0]
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def flatten_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3:
        return output[:, 0, :]
    if output.ndim > 2:
        return torch.flatten(output, start_dim=1)
    return output


def extract_embedding(
    model: torch.nn.Module,
    volume: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    volume = volume.to(device)
    if volume.is_floating_point():
        volume = volume.to(dtype=dtype)
    with torch.inference_mode():
        if not hasattr(model, "extract_vision_feats"):
            raise AttributeError("Pillar-0 model does not expose extract_vision_feats")
        output = model.extract_vision_feats({"breast_mr": volume})
    embedding = flatten_output(output_tensor(output)).squeeze(0)
    return embedding.detach().cpu().float().numpy()


def feature_record(embedding: np.ndarray) -> dict[str, float]:
    return {f"emb_{idx:04d}": float(value) for idx, value in enumerate(embedding)}


def output_paths(output_root: Path, crop_mode: str, input_role: str) -> tuple[Path, Path]:
    output_dir = output_root / MODEL_KEY / f"{crop_mode}_{input_role}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "patient_embeddings.csv", output_dir / "summary.json"


def load_existing_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    records = pd.read_csv(path).to_dict(orient="records")
    return [dict(record) for record in records]


def write_embeddings(path: Path, records: list[dict[str, object]]) -> pd.DataFrame:
    embeddings = pd.DataFrame.from_records(records)
    embeddings.to_csv(path, index=False)
    return embeddings


def extract_patient_embedding(
    row: pd.Series,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    volume, metadata = build_input_volume(
        row=row,
        input_role=args.input_role,
        crop_mode=args.crop_mode,
        args=args,
    )
    embedding = extract_embedding(model=model, volume=volume, device=device, dtype=dtype)
    record = {
        "patient_id": str(row["patient_id"]),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "pcr": int(row["pcr"]),
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "resample_spacing_mm": args.resample_spacing_mm,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "target_depth": args.target_depth,
        **metadata,
    }
    record.update(feature_record(embedding))
    return record


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.cache_dir = args.cache_dir.resolve()
    set_hf_cache(args.cache_dir)

    device = choose_device(args.device)
    dtype = choose_torch_dtype(args.torch_dtype, device)
    usable_flag = args.usable_flag or default_usable_flag(args.crop_mode)
    rows = load_manifest(args.manifest, usable_flag=usable_flag, split=args.split)
    rows = rows.sort_values(["split", "patient_id"]).reset_index(drop=True)
    rows = limit_rows(
        rows=rows,
        max_patients=args.max_patients,
        sample_mode=args.sample_mode,
        seed=args.seed,
    ).reset_index(drop=True)

    embeddings_path, summary_path = output_paths(args.output_root, args.crop_mode, args.input_role)
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
                f"{len(completed_patient_ids)} existing patient embeddings.",
                flush=True,
            )

    model = load_pillar0(device=device, dtype=dtype)

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
            records.append(extract_patient_embedding(row, model, args, device, dtype))
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
        except Exception as exc:  # noqa: BLE001 - keep extracting usable cases.
            failures.append({"patient_id": patient_id, "error": str(exc)})
            print(f"[{index + 1}/{len(rows)}] failed {patient_id}: {exc}", flush=True)

    if not records:
        raise RuntimeError(f"No Pillar-0 embeddings were extracted. Failures: {failures[:5]}")

    embeddings = write_embeddings(embeddings_path, records)

    summary = {
        "manifest": str(args.manifest),
        "embeddings": str(embeddings_path),
        "model_key": MODEL_KEY,
        "model_id": MODEL_ID,
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "usable_flag": usable_flag,
        "split": args.split,
        "max_patients": args.max_patients,
        "sample_mode": args.sample_mode,
        "resample_spacing_mm": args.resample_spacing_mm,
        "target_height": args.target_height,
        "target_width": args.target_width,
        "target_depth": args.target_depth,
        "device": str(device),
        "torch_dtype": str(dtype).replace("torch.", ""),
        "log_every": log_every,
        "checkpoint_every": checkpoint_every,
        "rows_requested": int(len(rows)),
        "rows_embedded": int(len(records)),
        "rows_newly_embedded": int(rows_newly_embedded),
        "rows_skipped_resume": int(rows_skipped_resume),
        "rows_failed": int(len(failures)),
        "failures": failures,
        "embedding_dim": int(sum(c.startswith("emb_") for c in embeddings.columns)),
        "source": "Pillar-0 3D breast MRI embeddings from 3-channel DCE volumes",
    }
    write_json(summary_path, summary)
    print(f"Wrote embeddings: {embeddings_path}")
    print(f"Wrote summary:    {summary_path}")


if __name__ == "__main__":
    main()
