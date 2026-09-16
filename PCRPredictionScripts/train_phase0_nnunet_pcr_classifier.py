#!/usr/bin/env python3
"""Train a phase-0 pCR classifier initialized from the MAMA-MIA nnU-Net encoder."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent / "outputs" / "phase0_nnunet_roi" / "roi_manifest.csv"
)
DEFAULT_NNUNET_MODEL_FOLDER = (
    PROJECT_ROOT
    / "ReconstructionScripts"
    / "qc_reports"
    / "mamamia_nnunet_qc_phase0_anchor"
    / "nnUNet_results"
    / "Dataset105_full_image"
    / "nnUNetTrainer__nnUNetPlans__3d_fullres"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "phase0_nnunet_pcr_classifier"
DEFAULT_CLINICAL_COLUMNS = [
    "age",
    "er",
    "pr",
    "her2",
    "hr",
    "tumor_subtype",
    "menopause",
    "nottingham_grade",
    "field_strength",
]


@dataclass
class NumericSpec:
    name: str
    kind: str
    mean: float
    std: float


@dataclass
class CategoricalSpec:
    name: str
    kind: str
    categories: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train/smoke-test a pCR classifier that reuses the pretrained "
            "MAMA-MIA nnU-Net encoder."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--nnunet-model-folder", type=Path, default=DEFAULT_NNUNET_MODEL_FOLDER)
    parser.add_argument("--checkpoint-name", default="checkpoint_final.pth")
    parser.add_argument("--fold", default="0", help="nnU-Net fold to initialize from.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--roi-status", nargs="+", default=("nnunet_pass",))
    parser.add_argument(
        "--architecture",
        choices=("nnunet_encoder", "swin3d_t", "r3d_18"),
        default="nnunet_encoder",
        help=(
            "Image model backbone. nnunet_encoder uses the MAMA-MIA nnU-Net "
            "encoder; swin3d_t and r3d_18 use torchvision 3D classification models."
        ),
    )
    parser.add_argument("--input-shape", type=int, nargs=3, default=(96, 96, 64))
    parser.add_argument(
        "--resize-mode",
        choices=("stretch", "pad"),
        default="stretch",
        help=(
            "How to map each crop to --input-shape. 'stretch' directly resizes "
            "each axis. 'pad' preserves the 3D voxel aspect ratio, then pads."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--final-test-checkpoint",
        choices=("best", "last"),
        default="best",
        help=(
            "Checkpoint loaded for the automatic final test evaluation. "
            "Use 'best' for validation-selected test reporting."
        ),
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument(
        "--random-init-encoder",
        action="store_true",
        help=(
            "Use the nnU-Net encoder architecture but reset its weights instead "
            "of using the pretrained MAMA-MIA segmentation weights."
        ),
    )
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--tabular-hidden-dim", type=int, default=64)
    parser.add_argument("--clinical-columns", nargs="*", default=None)
    parser.add_argument("--image-only", action="store_true")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision on CUDA/ROCm devices to reduce memory use.",
    )
    parser.add_argument("--wandb", action="store_true", help="Log training to Weights & Biases.")
    parser.add_argument(
        "--wandb-project",
        default="mamamia-phase1-pcr",
        help="Weights & Biases project name used when --wandb is enabled.",
    )
    parser.add_argument("--wandb-entity", default=None, help="Optional Weights & Biases entity.")
    parser.add_argument("--wandb-run-name", default=None, help="Optional Weights & Biases run name.")
    parser.add_argument("--wandb-group", default="phase1_mamamia_phase0", help="Optional W&B group.")
    parser.add_argument("--wandb-tags", nargs="*", default=None, help="Optional W&B tags.")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="Weights & Biases mode passed to wandb.init.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--max-test-batches", type=int, default=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    return series.fillna("").astype(str).str.lower().isin({"true", "1", "yes"})


def normalize_crop(crop: np.ndarray) -> np.ndarray:
    crop = crop.astype(np.float32, copy=False)
    finite = crop[np.isfinite(crop)]
    if finite.size == 0:
        return np.zeros_like(crop, dtype=np.float32)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    crop = np.clip(crop, lo, hi)
    mean = float(crop.mean())
    std = float(crop.std())
    if std < 1e-6:
        return crop * 0.0
    return ((crop - mean) / std).astype(np.float32, copy=False)


def crop_with_padding(image: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    shape = np.asarray(image.shape, dtype=int)
    out_shape = np.maximum(end - start, 1)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, shape)
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    fill_value = float(np.nanmedian(image)) if np.isfinite(image).any() else 0.0
    out = np.full(tuple(out_shape), fill_value, dtype=np.float32)
    if np.all(src_end > src_start):
        out[
            dst_start[0] : dst_end[0],
            dst_start[1] : dst_end[1],
            dst_start[2] : dst_end[2],
        ] = image[
            src_start[0] : src_end[0],
            src_start[1] : src_end[1],
            src_start[2] : src_end[2],
        ].astype(np.float32)
    return out


def resize_tensor(
    tensor: torch.Tensor,
    input_shape: tuple[int, int, int],
    resize_mode: str,
) -> torch.Tensor:
    if resize_mode == "stretch":
        return F.interpolate(tensor, size=input_shape, mode="trilinear", align_corners=False)

    if resize_mode != "pad":
        raise ValueError(f"Unsupported resize mode: {resize_mode}")

    source_shape = np.asarray(tensor.shape[2:], dtype=float)
    target_shape = np.asarray(input_shape, dtype=int)
    scale = float(np.min(target_shape / source_shape))
    resized_shape = np.round(source_shape * scale).astype(int)
    resized_shape = np.minimum(np.maximum(resized_shape, 1), target_shape)

    resized = F.interpolate(
        tensor,
        size=tuple(int(v) for v in resized_shape),
        mode="trilinear",
        align_corners=False,
    )
    pad_total = target_shape - resized_shape
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    pad = [
        int(pad_before[2]),
        int(pad_after[2]),
        int(pad_before[1]),
        int(pad_after[1]),
        int(pad_before[0]),
        int(pad_after[0]),
    ]
    return F.pad(resized, pad, mode="constant", value=0.0)


def fit_tabular_schema(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    schema: list[dict[str, Any]] = []
    for column in columns:
        if column not in df:
            print(f"Skipping missing clinical column: {column}")
            continue
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_fraction = float(numeric.notna().mean())
        if pd.api.types.is_numeric_dtype(series) or numeric_fraction >= 0.9:
            mean = float(numeric.mean()) if numeric.notna().any() else 0.0
            std = float(numeric.std()) if numeric.notna().sum() > 1 else 1.0
            if not np.isfinite(std) or std < 1e-6:
                std = 1.0
            schema.append(asdict(NumericSpec(column, "numeric", mean, std)))
        else:
            categories = (
                series.fillna("__missing__").astype(str).map(str.strip).replace("", "__missing__")
            )
            schema.append(
                asdict(CategoricalSpec(column, "categorical", sorted(categories.unique())))
            )
    return schema


def transform_tabular(df: pd.DataFrame, schema: list[dict[str, Any]]) -> np.ndarray:
    features: list[np.ndarray] = []
    for spec in schema:
        column = spec["name"]
        if spec["kind"] == "numeric":
            numeric = pd.to_numeric(df[column], errors="coerce").fillna(spec["mean"])
            values = ((numeric - spec["mean"]) / spec["std"]).to_numpy(dtype=np.float32)
            features.append(values[:, None])
        else:
            values = df[column].fillna("__missing__").astype(str).map(str.strip).replace("", "__missing__")
            cats = spec["categories"]
            one_hot = np.zeros((len(df), len(cats)), dtype=np.float32)
            cat_to_idx = {cat: idx for idx, cat in enumerate(cats)}
            missing_idx = cat_to_idx.get("__missing__")
            for row_idx, value in enumerate(values):
                col_idx = cat_to_idx.get(value, missing_idx)
                if col_idx is not None:
                    one_hot[row_idx, col_idx] = 1.0
            features.append(one_hot)
    if not features:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(features, axis=1).astype(np.float32)


class Phase0ROIDataset(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        tabular: np.ndarray,
        input_shape: tuple[int, int, int],
        resize_mode: str = "stretch",
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.tabular = tabular.astype(np.float32, copy=False)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.resize_mode = resize_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        row = self.rows.iloc[idx]
        image_path = resolve_path(row["reconstructed_image_path"])
        image = np.asanyarray(nib.load(str(image_path)).dataobj)
        start = np.asarray(
            [row["crop_start_x"], row["crop_start_y"], row["crop_start_z"]], dtype=int
        )
        end = np.asarray([row["crop_end_x"], row["crop_end_y"], row["crop_end_z"]], dtype=int)
        crop = normalize_crop(crop_with_padding(image, start, end))
        tensor = torch.from_numpy(crop[None, None]).float()
        tensor = resize_tensor(tensor, self.input_shape, self.resize_mode)
        tensor = tensor.squeeze(0)
        return {
            "image": tensor,
            "tabular": torch.from_numpy(self.tabular[idx]),
            "label": torch.tensor(float(row["pcr"]), dtype=torch.float32),
            "patient_id": str(row["patient_id"]),
        }


def reset_module_parameters(module: nn.Module) -> None:
    reset = getattr(module, "reset_parameters", None)
    if callable(reset):
        reset()


class NNUNetEncoderPCRClassifier(nn.Module):
    def __init__(
        self,
        model_folder: Path,
        fold: str,
        checkpoint_name: str,
        input_shape: tuple[int, int, int],
        tabular_dim: int,
        hidden_dim: int,
        tabular_hidden_dim: int,
        dropout: float,
        freeze_encoder: bool,
        random_init_encoder: bool,
    ) -> None:
        super().__init__()
        predictor = nnUNetPredictor(
            device=torch.device("cpu"),
            perform_everything_on_device=False,
            verbose=False,
            allow_tqdm=False,
        )
        predictor.initialize_from_trained_model_folder(
            str(model_folder), use_folds=(fold,), checkpoint_name=checkpoint_name
        )
        self.encoder = predictor.network.encoder
        if random_init_encoder:
            self.encoder.apply(reset_module_parameters)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        with torch.no_grad():
            was_training = self.encoder.training
            self.encoder.eval()
            dummy = torch.zeros((1, 1, *input_shape), dtype=torch.float32)
            encoded = self.encoder(dummy)
            image_dim = int(encoded[-1].shape[1] if isinstance(encoded, list) else encoded.shape[1])
            self.encoder.train(was_training)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.tabular_net: nn.Module | None
        if tabular_dim > 0:
            self.tabular_net = nn.Sequential(
                nn.Linear(tabular_dim, tabular_hidden_dim),
                nn.LayerNorm(tabular_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            fusion_dim = image_dim + tabular_hidden_dim
        else:
            self.tabular_net = None
            fusion_dim = image_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(image)
        image_features = encoded[-1] if isinstance(encoded, list) else encoded
        image_features = self.pool(image_features).flatten(1)
        if self.tabular_net is not None:
            if tabular is None:
                raise ValueError("Tabular features are required for this model")
            tabular_features = self.tabular_net(tabular)
            image_features = torch.cat([image_features, tabular_features], dim=1)
        return self.classifier(image_features).squeeze(1)


def one_channel_conv3d(conv: nn.Conv3d) -> nn.Conv3d:
    new_conv = nn.Conv3d(
        in_channels=1,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=1,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        new_conv.weight.copy_(conv.weight.mean(dim=1, keepdim=True))
        if conv.bias is not None and new_conv.bias is not None:
            new_conv.bias.copy_(conv.bias)
    return new_conv


class TorchvisionVideoPCRClassifier(nn.Module):
    def __init__(self, architecture: str, tabular_dim: int) -> None:
        super().__init__()
        if tabular_dim > 0:
            raise ValueError(
                f"{architecture} is currently supported for image-only runs. "
                "Use --image-only for this architecture."
            )

        if architecture == "swin3d_t":
            from torchvision.models.video import swin3d_t

            self.model = swin3d_t(weights=None, num_classes=1)
            self.model.patch_embed.proj = one_channel_conv3d(self.model.patch_embed.proj)
        elif architecture == "r3d_18":
            from torchvision.models.video import r3d_18

            self.model = r3d_18(weights=None, num_classes=1)
            self.model.stem[0] = one_channel_conv3d(self.model.stem[0])
        else:
            raise ValueError(f"Unsupported torchvision video architecture: {architecture}")

    def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None) -> torch.Tensor:
        del tabular
        video_tensor = image.permute(0, 1, 4, 2, 3).contiguous()
        return self.model(video_tensor).squeeze(1)


def build_pcr_model(
    architecture: str,
    model_folder: Path,
    fold: str,
    checkpoint_name: str,
    input_shape: tuple[int, int, int],
    tabular_dim: int,
    hidden_dim: int,
    tabular_hidden_dim: int,
    dropout: float,
    freeze_encoder: bool,
    random_init_encoder: bool,
) -> nn.Module:
    if architecture == "nnunet_encoder":
        return NNUNetEncoderPCRClassifier(
            model_folder=model_folder,
            fold=fold,
            checkpoint_name=checkpoint_name,
            input_shape=input_shape,
            tabular_dim=tabular_dim,
            hidden_dim=hidden_dim,
            tabular_hidden_dim=tabular_hidden_dim,
            dropout=dropout,
            freeze_encoder=freeze_encoder,
            random_init_encoder=random_init_encoder,
        )

    if freeze_encoder:
        raise ValueError(f"--freeze-encoder is only supported with --architecture nnunet_encoder")
    if random_init_encoder:
        raise ValueError(
            f"--random-init-encoder is only meaningful with --architecture nnunet_encoder; "
            f"{architecture} already starts from random weights."
        )
    return TorchvisionVideoPCRClassifier(architecture=architecture, tabular_dim=tabular_dim)


def prepare_splits(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_pool = df[df["split"].fillna("train") == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    if train_pool.empty:
        raise ValueError("No training rows found in manifest")

    stratify = train_pool["pcr"] if train_pool["pcr"].nunique() == 2 else None
    train_df, val_df = train_test_split(
        train_pool,
        test_size=val_frac,
        random_state=seed,
        stratify=stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def make_loader(
    rows: pd.DataFrame,
    tabular: np.ndarray,
    input_shape: tuple[int, int, int],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    resize_mode: str,
) -> DataLoader:
    dataset = Phase0ROIDataset(rows, tabular, input_shape, resize_mode=resize_mode)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def compute_metrics(labels: list[float], logits: list[float]) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.float32)
    z = np.asarray(logits, dtype=np.float32)
    probs = 1.0 / (1.0 + np.exp(-z))
    pred = probs >= 0.5
    if len(np.unique(y)) == 2:
        balanced_accuracy = float(balanced_accuracy_score(y.astype(int), pred.astype(int)))
        roc_auc = float(roc_auc_score(y, probs))
        average_precision = float(average_precision_score(y, probs))
    else:
        balanced_accuracy = float("nan")
        roc_auc = float("nan")
        average_precision = float("nan")
    metrics = {
        "loss": float("nan"),
        "balanced_accuracy": balanced_accuracy,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
    }
    return metrics


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [json_ready(v) for v in value]
    if isinstance(value, list):
        return [json_ready(v) for v in value]
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_batches(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    amp_enabled = bool(use_amp and device.type == "cuda")
    model.train(training)
    total_loss = 0.0
    total_examples = 0
    labels: list[float] = []
    logits_out: list[float] = []

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        image = batch["image"].to(device, non_blocking=True)
        tabular = batch["tabular"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(image, tabular if tabular.shape[1] else None)
                loss = criterion(logits, label)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        batch_size = int(label.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        labels.extend(label.detach().cpu().numpy().tolist())
        logits_out.extend(logits.detach().cpu().numpy().tolist())

    metrics = compute_metrics(labels, logits_out)
    metrics["loss"] = total_loss / max(total_examples, 1)
    return metrics


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    tabular_schema: list[dict[str, Any]],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "tabular_schema": tabular_schema,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint_for_final_test(
    model: nn.Module,
    output_dir: Path,
    checkpoint_name: str,
) -> dict[str, Any]:
    checkpoint_path = output_dir / f"{checkpoint_name}.pt"
    if not checkpoint_path.exists() and checkpoint_name == "best":
        fallback = output_dir / "last.pt"
        if fallback.exists():
            print(
                "WARNING: best.pt was not found, so final test evaluation will "
                "use last.pt instead."
            )
            checkpoint_path = fallback
    if not checkpoint_path.exists():
        return {
            "checkpoint": "in_memory",
            "checkpoint_epoch": None,
            "checkpoint_val_metrics": {},
        }

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return {
        "checkpoint": checkpoint_path.name,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_val_metrics": checkpoint.get("metrics", {}),
    }


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def init_wandb_run(
    args: argparse.Namespace,
    split_summary: dict[str, Any],
    tabular_schema: list[dict[str, Any]],
):
    if not args.wandb:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "Weights & Biases logging was requested with --wandb, but wandb is "
            "not installed. Install it with ./.venv/bin/pip install wandb"
        ) from exc

    tags = list(args.wandb_tags or [])
    if args.architecture not in tags:
        tags.append(args.architecture)
    if args.image_only and "image-only" not in tags:
        tags.append("image-only")

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.output_dir.name,
        group=args.wandb_group,
        tags=tags,
        mode=args.wandb_mode,
        config=json_ready(
            {
                "args": vars(args),
                "split_summary": split_summary,
                "tabular_schema": tabular_schema,
            }
        ),
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    usable = manifest[
        as_bool(manifest["usable_for_training"])
        & manifest["roi_status"].isin(args.roi_status)
        & manifest["pcr"].notna()
    ].copy()
    usable["pcr"] = usable["pcr"].astype(int)
    if usable.empty:
        raise ValueError("No usable training rows found. Check ROI status filters.")

    train_df, val_df, test_df = prepare_splits(usable, args.val_frac, args.seed)
    clinical_columns = [] if args.image_only else (args.clinical_columns or DEFAULT_CLINICAL_COLUMNS)
    tabular_schema = fit_tabular_schema(train_df, clinical_columns)
    train_tab = transform_tabular(train_df, tabular_schema)
    val_tab = transform_tabular(val_df, tabular_schema)
    test_tab = transform_tabular(test_df, tabular_schema) if not test_df.empty else np.zeros((0, 0), dtype=np.float32)

    input_shape = tuple(int(v) for v in args.input_shape)
    train_loader = make_loader(
        train_df,
        train_tab,
        input_shape,
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
        args.resize_mode,
    )
    val_loader = make_loader(
        val_df,
        val_tab,
        input_shape,
        args.batch_size,
        False,
        args.num_workers,
        args.seed,
        args.resize_mode,
    )

    device = torch.device(args.device)
    model = build_pcr_model(
        architecture=args.architecture,
        model_folder=args.nnunet_model_folder,
        fold=args.fold,
        checkpoint_name=args.checkpoint_name,
        input_shape=input_shape,
        tabular_dim=train_tab.shape[1],
        hidden_dim=args.hidden_dim,
        tabular_hidden_dim=args.tabular_hidden_dim,
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder,
        random_init_encoder=args.random_init_encoder,
    ).to(device)

    pos = float((train_df["pcr"] == 1).sum())
    neg = float((train_df["pcr"] == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    split_summary = {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_pcr_counts": {
            str(int(k)): int(v) for k, v in train_df["pcr"].value_counts().sort_index().items()
        },
        "val_pcr_counts": {
            str(int(k)): int(v) for k, v in val_df["pcr"].value_counts().sort_index().items()
        },
        "test_pcr_counts": {
            str(int(k)): int(v) for k, v in test_df["pcr"].value_counts().sort_index().items()
        },
        "clinical_columns": clinical_columns,
        "tabular_dim": int(train_tab.shape[1]),
        "architecture": args.architecture,
        "input_shape": list(input_shape),
        "resize_mode": args.resize_mode,
        "freeze_encoder": bool(args.freeze_encoder),
        "random_init_encoder": bool(args.random_init_encoder),
        "amp": amp_enabled,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "args": json_ready(vars(args)),
                "split_summary": split_summary,
                "tabular_schema": tabular_schema,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps(split_summary, indent=2))

    wandb_run = init_wandb_run(args, split_summary, tabular_schema)
    try:
        if args.dry_run:
            metrics = run_batches(
                model,
                train_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                use_amp=args.amp,
                max_batches=1,
            )
            if wandb_run is not None:
                wandb_run.log(prefixed_metrics("dry_run", metrics), step=0)
            print("Dry-run forward pass OK")
            print(json.dumps(metrics, indent=2, sort_keys=True))
            return

        metrics_path = args.output_dir / "metrics.csv"
        best_auc = -np.inf
        with metrics_path.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_balanced_accuracy",
                    "train_roc_auc",
                    "train_average_precision",
                    "val_loss",
                    "val_balanced_accuracy",
                    "val_roc_auc",
                    "val_average_precision",
                ],
            )
            writer.writeheader()
            for epoch in range(1, args.epochs + 1):
                train_metrics = run_batches(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    scaler=scaler,
                    use_amp=args.amp,
                    max_batches=args.max_train_batches,
                )
                val_metrics = run_batches(
                    model,
                    val_loader,
                    criterion,
                    device,
                    optimizer=None,
                    scaler=None,
                    use_amp=args.amp,
                    max_batches=args.max_val_batches,
                )
                row = {
                    "epoch": epoch,
                    "train_loss": train_metrics["loss"],
                    "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                    "train_roc_auc": train_metrics["roc_auc"],
                    "train_average_precision": train_metrics["average_precision"],
                    "val_loss": val_metrics["loss"],
                    "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_average_precision": val_metrics["average_precision"],
                }
                writer.writerow(row)
                f.flush()
                print(json.dumps(row, sort_keys=True))

                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "epoch": epoch,
                            **prefixed_metrics("train", train_metrics),
                            **prefixed_metrics("val", val_metrics),
                        },
                        step=epoch,
                    )

                score = val_metrics["roc_auc"]
                if np.isfinite(score) and score > best_auc:
                    best_auc = score
                    save_checkpoint(
                        args.output_dir / "best.pt",
                        model,
                        optimizer,
                        args,
                        tabular_schema,
                        epoch,
                        val_metrics,
                    )
                    if wandb_run is not None:
                        wandb_run.summary["best/epoch"] = epoch
                        for key, value in val_metrics.items():
                            wandb_run.summary[f"best/val/{key}"] = value
                save_checkpoint(
                    args.output_dir / "last.pt",
                    model,
                    optimizer,
                    args,
                    tabular_schema,
                    epoch,
                    val_metrics,
                )

        if not test_df.empty:
            test_checkpoint = load_checkpoint_for_final_test(
                model,
                args.output_dir,
                args.final_test_checkpoint,
            )
            test_loader = make_loader(
                test_df,
                test_tab,
                input_shape,
                args.batch_size,
                False,
                args.num_workers,
                args.seed,
                args.resize_mode,
            )
            test_metrics = run_batches(
                model,
                test_loader,
                criterion,
                device,
                optimizer=None,
                scaler=None,
                use_amp=args.amp,
                max_batches=args.max_test_batches,
            )
            (args.output_dir / "test_metrics.json").write_text(
                json.dumps(test_metrics, indent=2) + "\n"
            )
            (args.output_dir / "test_evaluation.json").write_text(
                json.dumps(
                    json_ready(
                        {
                            **test_checkpoint,
                            "split": "test",
                            "rows": int(len(test_df)),
                            "pcr_counts": {
                                str(int(k)): int(v)
                                for k, v in test_df["pcr"].value_counts().sort_index().items()
                            },
                            "metrics": test_metrics,
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            if wandb_run is not None:
                wandb_run.log(prefixed_metrics("test", test_metrics), step=args.epochs + 1)
                wandb_run.summary["test/checkpoint"] = test_checkpoint["checkpoint"]
                wandb_run.summary["test/checkpoint_epoch"] = test_checkpoint["checkpoint_epoch"]
                for key, value in test_metrics.items():
                    wandb_run.summary[f"test/{key}"] = value
            print("test_metrics")
            print(json.dumps(test_metrics, indent=2, sort_keys=True))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
