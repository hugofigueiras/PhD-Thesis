#!/usr/bin/env python3
"""Extract patient-level embeddings from 2D foundation models.

This extractor treats each 3D breast MRI volume as a stack of 2D slices,
embeds sampled axial slices, and aggregates slice embeddings into one
patient-level vector for the logistic probe.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent
    / "outputs"
    / "manifests"
    / "mamamia_multiphase_foundation_manifest.csv"
)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "outputs" / "embeddings"
DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "outputs" / "model_cache"

MODEL_SPECS = {
    "biomedclip": {
        "model_id": "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "loader": "open_clip",
        "image_size": 224,
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
        "embedding_note": "Biomedical CLIP ViT-B/16 image features pretrained on PMC-15M.",
    },
    "curia": {
        "model_id": "raidium/curia",
        "loader": "transformers_processor",
        "processor_class": "auto_image_processor",
        "processor_image_format": "array",
        "image_size": None,
        "mean": None,
        "std": None,
        "embedding_note": "Radiology DINOv2-style model trained on cross-sectional imaging exams.",
    },
    "medsiglip": {
        "model_id": "google/medsiglip-448",
        "loader": "transformers_processor",
        "processor_class": "auto_image_processor",
        "processor_image_format": "pil_rgb",
        "image_size": None,
        "mean": None,
        "std": None,
        "embedding_note": (
            "MedSigLIP 448px medical image-text encoder trained on medical "
            "image-text pairs plus natural images."
        ),
    },
    "radiodino": {
        "model_id": "hf_hub:Snarcy/RadioDino-s16",
        "loader": "timm",
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "embedding_note": "ViT-small DINO features pretrained on RadImageNet.",
    }
}

SUBTRACTION_ROLES = {
    "phase1_minus_phase0": ("phase1_path", "phase0_path"),
    "phase2_minus_phase0": ("phase2_path", "phase0_path"),
    "last_phase_minus_phase0": ("last_phase_path", "phase0_path"),
}
PHASE_ROLES = {
    "phase0": "phase0_path",
    "phase1": "phase1_path",
    "phase2": "phase2_path",
    "last_phase": "last_phase_path",
}


@dataclass
class LoadedModel:
    model: torch.nn.Module
    loader: str
    processor: object | None = None
    dtype: torch.dtype | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract patient-level embeddings from sampled 2D slices of "
            "MAMA-MIA multiphase MRI volumes."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-key", choices=sorted(MODEL_SPECS), default="radiodino")
    parser.add_argument(
        "--input-role",
        choices=sorted([*PHASE_ROLES, *SUBTRACTION_ROLES]),
        default="phase0",
    )
    parser.add_argument("--crop-mode", choices=("whole", "expert_roi"), default="whole")
    parser.add_argument(
        "--aggregation",
        choices=("mean", "max", "mean_max"),
        default="mean",
        help="How to aggregate slice embeddings into patient-level embeddings.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--usable-flag", default=None)
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument(
        "--sample-mode",
        choices=("stratified", "head"),
        default="stratified",
        help=(
            "How to choose rows when --max-patients is set. Stratified keeps a "
            "mix of official train/test and pCR labels when possible."
        ),
    )
    parser.add_argument("--max-slices", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--torch-dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help=(
            "Torch dtype for Transformers processor models. Auto uses float16 on "
            "CUDA and float32 on CPU."
        ),
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print progress every N patients; failures and the final patient always print.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Load architecture without pretrained weights for script debugging only.",
    )
    return parser.parse_args()


def set_hf_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("HF_HUB_CACHE", str((cache_dir / "hub").resolve()))


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def choose_torch_dtype(
    value: str,
    device: torch.device,
    loader: str,
) -> torch.dtype | None:
    if loader != "transformers_processor":
        return None
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported torch dtype: {value}")


def load_manifest(path: Path, usable_flag: str | None, split: str) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    manifest = manifest.copy()
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
        raise ValueError("No eligible rows found for embedding extraction")
    return rows


def limit_rows(rows: pd.DataFrame, max_patients: int | None, sample_mode: str, seed: int) -> pd.DataFrame:
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


def default_usable_flag(crop_mode: str) -> str:
    if crop_mode == "expert_roi":
        return "usable_for_expert_roi_benchmark"
    return "usable_for_whole_volume_benchmark"


def load_model(
    model_key: str,
    pretrained: bool,
    device: torch.device,
    torch_dtype: torch.dtype | None,
) -> LoadedModel:
    spec = MODEL_SPECS[model_key]
    processor = None
    if spec["loader"] == "timm":
        import timm

        try:
            model = timm.create_model(
                str(spec["model_id"]),
                pretrained=pretrained,
                num_classes=0,
            )
        except TypeError:
            model = timm.create_model(str(spec["model_id"]), pretrained=pretrained)
    elif spec["loader"] == "open_clip":
        if not pretrained:
            raise ValueError("--no-pretrained is not supported for OpenCLIP models")
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "OpenCLIP models require open_clip_torch. Install it with: "
                "./.venv/bin/pip install open_clip_torch transformers"
            ) from exc

        model, _ = open_clip.create_model_from_pretrained(str(spec["model_id"]))
    elif spec["loader"] == "transformers_processor":
        if not pretrained:
            raise ValueError("--no-pretrained is not supported for Transformers processor models")
        try:
            from transformers import AutoImageProcessor, AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "Transformers processor models require transformers. Install it with: "
                "./.venv/bin/pip install transformers"
            ) from exc

        processor_class = str(spec.get("processor_class", "auto_image_processor"))
        if processor_class == "auto_processor":
            processor = AutoProcessor.from_pretrained(
                str(spec["model_id"]),
                trust_remote_code=True,
            )
        elif processor_class == "auto_image_processor":
            processor = AutoImageProcessor.from_pretrained(
                str(spec["model_id"]),
                trust_remote_code=True,
            )
        else:
            raise ValueError(
                f"Unsupported processor class for {model_key}: {processor_class}"
            )
        model_kwargs = {"trust_remote_code": True}
        if torch_dtype is not None:
            model_kwargs["dtype"] = torch_dtype
        try:
            model = AutoModel.from_pretrained(str(spec["model_id"]), **model_kwargs)
        except TypeError:
            if "dtype" not in model_kwargs:
                raise
            model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
            model = AutoModel.from_pretrained(str(spec["model_id"]), **model_kwargs)
    else:
        raise ValueError(f"Unsupported loader for {model_key}: {spec['loader']}")

    model.eval()
    model.to(device)
    return LoadedModel(
        model=model,
        loader=str(spec["loader"]),
        processor=processor,
        dtype=torch_dtype,
    )


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


def load_volume(row: pd.Series, input_role: str, crop_mode: str) -> np.ndarray:
    crop = crop_slices(row, crop_mode)
    if input_role in PHASE_ROLES:
        path = path_value(row, PHASE_ROLES[input_role])
        image = nib.load(str(path))
        return np.asanyarray(image.dataobj[crop]).astype(np.float32)

    if input_role in SUBTRACTION_ROLES:
        post_column, base_column = SUBTRACTION_ROLES[input_role]
        post = np.asanyarray(nib.load(str(path_value(row, post_column))).dataobj[crop]).astype(
            np.float32
        )
        base = np.asanyarray(nib.load(str(path_value(row, base_column))).dataobj[crop]).astype(
            np.float32
        )
        if post.shape != base.shape:
            raise ValueError(f"Subtraction shape mismatch: {post.shape} vs {base.shape}")
        return post - base

    raise ValueError(f"Unsupported input role: {input_role}")


def sampled_slice_indices(z_size: int, max_slices: int) -> np.ndarray:
    if z_size <= 0:
        raise ValueError("Volume has no z slices")
    count = min(int(max_slices), int(z_size))
    if count <= 0:
        raise ValueError("max_slices must be positive")
    if count == z_size:
        return np.arange(z_size, dtype=int)
    return np.unique(np.linspace(0, z_size - 1, count).round().astype(int))


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    values = np.asarray(slice_2d, dtype=np.float32)
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


def slices_to_tensor(
    volume: np.ndarray,
    slice_indices: Iterable[int],
    image_size: int,
    mean: Iterable[float],
    std: Iterable[float],
) -> torch.Tensor:
    slices = []
    for z_index in slice_indices:
        normalized = normalize_slice(volume[:, :, int(z_index)])
        tensor = torch.from_numpy(normalized).unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)
        slices.append(tensor)

    batch = torch.stack(slices, dim=0).float()
    batch = F.interpolate(
        batch,
        size=(int(image_size), int(image_size)),
        mode="bilinear",
        align_corners=False,
    )
    mean_tensor = torch.tensor(tuple(mean), dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(tuple(std), dtype=torch.float32).view(1, 3, 1, 1)
    return (batch - mean_tensor) / std_tensor


def slices_to_arrays(volume: np.ndarray, slice_indices: Iterable[int]) -> list[np.ndarray]:
    return [normalize_slice(volume[:, :, int(z_index)]) for z_index in slice_indices]


def slices_to_pil_rgb(volume: np.ndarray, slice_indices: Iterable[int]) -> list[object]:
    from PIL import Image

    images = []
    for z_index in slice_indices:
        normalized = normalize_slice(volume[:, :, int(z_index)])
        uint8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
        rgb = np.repeat(uint8[:, :, None], 3, axis=2)
        images.append(Image.fromarray(rgb, mode="RGB"))
    return images


def tensor_to_device(
    value: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    value = value.to(device)
    if dtype is not None and value.is_floating_point():
        value = value.to(dtype=dtype)
    return value


def inputs_to_device(inputs, device: torch.device, dtype: torch.dtype | None = None):
    if isinstance(inputs, dict):
        return {
            key: tensor_to_device(value, device, dtype) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
    if hasattr(inputs, "items"):
        return {
            key: tensor_to_device(value, device, dtype) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return inputs


def process_transformers_images(processor, images: list[object]):
    try:
        return processor(images=images, return_tensors="pt")
    except TypeError:
        return processor(images, return_tensors="pt")
    except Exception as batch_error:
        individual_inputs = []
        for image in images:
            try:
                try:
                    individual_inputs.append(processor(images=image, return_tensors="pt"))
                except TypeError:
                    individual_inputs.append(processor(image, return_tensors="pt"))
            except Exception:
                raise batch_error

        merged = {}
        for key in individual_inputs[0].keys():
            values = [item[key] for item in individual_inputs]
            if torch.is_tensor(values[0]):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = values
        return merged


def output_tensor(output) -> torch.Tensor:
    for attribute in ("image_embeds", "pooler_output", "last_hidden_state"):
        value = getattr(output, attribute, None)
        if value is not None:
            return value

    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output", "last_hidden_state"):
            value = output.get(key)
            if value is not None:
                return value

    if isinstance(output, (tuple, list)):
        return output[0]

    if torch.is_tensor(output):
        return output

    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def flatten_output(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 3:
        return output[:, 0, :]
    if output.ndim > 2:
        return torch.flatten(output, start_dim=1)
    return output


def model_forward_embeddings(
    loaded_model: LoadedModel,
    images: torch.Tensor | list[object],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    model = loaded_model.model

    if loaded_model.loader == "transformers_processor":
        if loaded_model.processor is None:
            raise ValueError("Transformers processor model is missing its processor")
        if not isinstance(images, list):
            raise TypeError("Transformers processor models expect a list of 2D numpy slices")

        for start in range(0, len(images), batch_size):
            batch_images = images[start : start + batch_size]
            processor_inputs = process_transformers_images(
                loaded_model.processor,
                batch_images,
            )
            processor_inputs = inputs_to_device(
                processor_inputs,
                device,
                loaded_model.dtype,
            )
            with torch.inference_mode():
                if hasattr(model, "get_image_features") and "pixel_values" in processor_inputs:
                    output = model.get_image_features(
                        pixel_values=processor_inputs["pixel_values"]
                    )
                else:
                    output = model(**processor_inputs)
            outputs.append(flatten_output(output_tensor(output)).detach().cpu().float().numpy())
        return np.concatenate(outputs, axis=0)

    if not torch.is_tensor(images):
        raise TypeError("Tensor-based models expect a torch.Tensor image batch")

    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size].to(device)
        with torch.inference_mode():
            if hasattr(model, "encode_image"):
                try:
                    output = model.encode_image(batch, normalize=False)
                except TypeError:
                    output = model.encode_image(batch)
            else:
                output = model(batch)
        outputs.append(flatten_output(output_tensor(output)).detach().cpu().float().numpy())
    return np.concatenate(outputs, axis=0)


def aggregate_embeddings(slice_embeddings: np.ndarray, aggregation: str) -> np.ndarray:
    if aggregation == "mean":
        return slice_embeddings.mean(axis=0)
    if aggregation == "max":
        return slice_embeddings.max(axis=0)
    if aggregation == "mean_max":
        return np.concatenate([slice_embeddings.mean(axis=0), slice_embeddings.max(axis=0)])
    raise ValueError(f"Unsupported aggregation: {aggregation}")


def output_paths(output_root: Path, model_key: str, crop_mode: str, input_role: str) -> tuple[Path, Path]:
    output_dir = output_root / model_key / f"{crop_mode}_{input_role}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / "patient_embeddings.csv", output_dir / "summary.json"


def feature_record(embedding: np.ndarray) -> dict[str, float]:
    return {f"emb_{idx:04d}": float(value) for idx, value in enumerate(embedding)}


def extract_patient_embedding(
    row: pd.Series,
    loaded_model: LoadedModel,
    spec: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, object]:
    volume = load_volume(row, args.input_role, args.crop_mode)
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")
    z_indices = sampled_slice_indices(volume.shape[2], args.max_slices)
    if spec["loader"] == "transformers_processor":
        processor_image_format = str(spec.get("processor_image_format", "array"))
        if processor_image_format == "array":
            images = slices_to_arrays(volume=volume, slice_indices=z_indices)
        elif processor_image_format == "pil_rgb":
            images = slices_to_pil_rgb(volume=volume, slice_indices=z_indices)
        else:
            raise ValueError(
                f"Unsupported processor image format: {processor_image_format}"
            )
    else:
        images = slices_to_tensor(
            volume=volume,
            slice_indices=z_indices,
            image_size=int(spec["image_size"]),
            mean=spec["mean"],
            std=spec["std"],
        )
    slice_embeddings = model_forward_embeddings(
        loaded_model=loaded_model,
        images=images,
        device=device,
        batch_size=args.batch_size,
    )
    patient_embedding = aggregate_embeddings(slice_embeddings, args.aggregation)

    record = {
        "patient_id": str(row["patient_id"]),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "pcr": int(row["pcr"]),
        "model_key": args.model_key,
        "model_id": spec["model_id"],
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "aggregation": args.aggregation,
        "num_slices": int(len(z_indices)),
        "slice_indices": ";".join(str(int(v)) for v in z_indices),
        "source_shape_x": int(volume.shape[0]),
        "source_shape_y": int(volume.shape[1]),
        "source_shape_z": int(volume.shape[2]),
    }
    record.update(feature_record(patient_embedding))
    return record


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    args.cache_dir = args.cache_dir.resolve()
    set_hf_cache(args.cache_dir)
    device = choose_device(args.device)
    usable_flag = args.usable_flag or default_usable_flag(args.crop_mode)
    rows = load_manifest(args.manifest, usable_flag=usable_flag, split=args.split)
    rows = rows.sort_values(["split", "patient_id"]).reset_index(drop=True)
    rows = limit_rows(
        rows=rows,
        max_patients=args.max_patients,
        sample_mode=args.sample_mode,
        seed=args.seed,
    ).reset_index(drop=True)

    spec = MODEL_SPECS[args.model_key]
    loaded_model = load_model(
        model_key=args.model_key,
        pretrained=not args.no_pretrained,
        device=device,
        torch_dtype=choose_torch_dtype(
            value=args.torch_dtype,
            device=device,
            loader=str(spec["loader"]),
        ),
    )

    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    log_every = max(1, int(args.log_every))
    for index, row in rows.iterrows():
        patient_id = str(row["patient_id"])
        try:
            records.append(extract_patient_embedding(row, loaded_model, spec, args, device))
            if (index + 1) % log_every == 0 or index + 1 == len(rows):
                print(f"[{index + 1}/{len(rows)}] embedded {patient_id}")
        except Exception as exc:  # noqa: BLE001 - continue and report bad cases.
            failures.append({"patient_id": patient_id, "error": str(exc)})
            print(f"[{index + 1}/{len(rows)}] failed {patient_id}: {exc}")

    if not records:
        raise RuntimeError(f"No embeddings were extracted. Failures: {failures[:5]}")

    embeddings_path, summary_path = output_paths(
        args.output_root,
        args.model_key,
        args.crop_mode,
        args.input_role,
    )
    embeddings = pd.DataFrame.from_records(records)
    embeddings.to_csv(embeddings_path, index=False)

    summary = {
        "manifest": str(args.manifest),
        "embeddings": str(embeddings_path),
        "model_key": args.model_key,
        "model_id": spec["model_id"],
        "loader": spec["loader"],
        "embedding_note": spec["embedding_note"],
        "input_role": args.input_role,
        "crop_mode": args.crop_mode,
        "aggregation": args.aggregation,
        "usable_flag": usable_flag,
        "split": args.split,
        "max_patients": args.max_patients,
        "sample_mode": args.sample_mode,
        "max_slices": args.max_slices,
        "batch_size": args.batch_size,
        "device": str(device),
        "torch_dtype": str(loaded_model.dtype).replace("torch.", "")
        if loaded_model.dtype is not None
        else None,
        "log_every": log_every,
        "pretrained": not args.no_pretrained,
        "rows_requested": int(len(rows)),
        "rows_embedded": int(len(records)),
        "rows_failed": int(len(failures)),
        "failures": failures,
        "embedding_dim": int(sum(c.startswith("emb_") for c in embeddings.columns)),
        "source": "2D slice embeddings aggregated per patient",
    }
    write_json(summary_path, summary)
    print(f"Wrote embeddings: {embeddings_path}")
    print(f"Wrote summary:    {summary_path}")


if __name__ == "__main__":
    main()
