from __future__ import annotations

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QC_ROOT = (
    PROJECT_ROOT / "ReconstructionScripts" / "qc_reports" / "mamamia_nnunet_qc"
)
DEFAULT_MANIFEST = DEFAULT_QC_ROOT / "qc_manifest.csv"
DEFAULT_PREDICTION_ROOT = DEFAULT_QC_ROOT / "nnunet_predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare nnU-Net predictions on reconstructed volumes with the original "
            "MAMA-MIA image/mask pair identified by patient, phase, and anchor date."
        )
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prediction-root", type=Path, default=DEFAULT_PREDICTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_QC_ROOT)
    parser.add_argument(
        "--mask-kind",
        choices=("expert", "automatic"),
        default="expert",
        help="MAMA-MIA mask to use as the trusted original reference.",
    )
    parser.add_argument(
        "--patients",
        nargs="+",
        default=None,
        help="Optional MAMA-MIA patient IDs to include.",
    )
    parser.add_argument(
        "--anchor-only",
        action="store_true",
        help="Only report reconstructed time points that match the MAMA-MIA acquisition date.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Limit reconstructed rows processed. Useful for smoke checks.",
    )
    parser.add_argument(
        "--make-overlays",
        action="store_true",
        help="Write side-by-side PNG overlays for visual QC.",
    )
    parser.add_argument(
        "--overlay-all-timepoints",
        action="store_true",
        help="Overlay all reconstructed time points instead of anchor matches only.",
    )
    parser.add_argument(
        "--max-overlays",
        type=int,
        default=50,
        help="Maximum number of overlay PNGs to write.",
    )
    return parser.parse_args()


def prediction_path(row: pd.Series, prediction_root: Path) -> Path:
    subdir = "mamamia_phases" if row["collection"] == "mamamia" else "reconstructed_phases"
    return prediction_root / subdir / f"{row['case_id']}.nii.gz"


def original_prediction_path(row: pd.Series, prediction_root: Path) -> Path:
    return prediction_root / "mamamia_phases" / f"{row['paired_mamamia_case_id']}.nii.gz"


def load_image(path: str | Path) -> nib.Nifti1Image:
    return nib.load(str(path))


def load_array(path: str | Path) -> np.ndarray:
    return np.asanyarray(load_image(path).dataobj)


def load_mask(path: str | Path) -> np.ndarray:
    return load_array(path) > 0


def same_grid(first: nib.Nifti1Image, second: nib.Nifti1Image) -> bool:
    return first.shape == second.shape and np.allclose(first.affine, second.affine, atol=1e-3)


def dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    denom = int(a.sum()) + int(b.sum())
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / denom)


def voxel_volume_ml(img: nib.Nifti1Image) -> float:
    return float(abs(np.linalg.det(img.affine[:3, :3])) / 1000.0)


def mask_volume_ml(mask: np.ndarray, img: nib.Nifti1Image) -> float:
    return float(mask.sum() * voxel_volume_ml(img))


def header_shape(img: nib.Nifti1Image) -> str:
    return "x".join(str(dim) for dim in img.shape)


def header_spacing(img: nib.Nifti1Image) -> str:
    zooms = img.header.get_zooms()[: len(img.shape)]
    return "x".join(f"{zoom:.6g}" for zoom in zooms)


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def summarize_row(row: pd.Series, prediction_root: Path, mask_kind: str) -> dict[str, object]:
    pred_path = prediction_path(row, prediction_root)
    original_pred_path = original_prediction_path(row, prediction_root)
    original_image_path = str(row.get("paired_mamamia_image_path", "") or "")
    original_mask_path = str(row.get(f"{mask_kind}_mask_path", "") or "")

    out: dict[str, object] = {
        "patient_id": row["patient_id"],
        "dataset": row["dataset"],
        "exam_date": row["exam_date"],
        "anchor_reconstructed_date": row["anchor_reconstructed_date"],
        "is_anchor_date": row["is_anchor_date"],
        "phase_index": row["phase_index"],
        "reconstructed_case_id": row["case_id"],
        "reconstructed_image_path": row["source_image_path"],
        "reconstructed_prediction_path": str(pred_path) if pred_path.exists() else "",
        "mamamia_case_id": row["paired_mamamia_case_id"],
        "mamamia_image_path": original_image_path,
        "mamamia_mask_kind": mask_kind,
        "mamamia_mask_path": original_mask_path,
        "mamamia_prediction_path": str(original_pred_path) if original_pred_path.exists() else "",
        "timepoint_match_method": (
            "mamamia_acquisition_date" if is_true(row["is_anchor_date"]) else "other_timepoint"
        ),
        "comparison_note": "",
    }

    try:
        recon_img = load_image(row["source_image_path"])
        out["reconstructed_shape"] = header_shape(recon_img)
        out["reconstructed_spacing"] = header_spacing(recon_img)
    except Exception as exc:
        out["comparison_note"] = f"could_not_load_reconstructed_image:{exc}"
        return out

    if original_image_path:
        try:
            mamamia_img = load_image(original_image_path)
            out["mamamia_shape"] = header_shape(mamamia_img)
            out["mamamia_spacing"] = header_spacing(mamamia_img)
            out["image_same_grid"] = same_grid(recon_img, mamamia_img)
        except Exception as exc:
            out["comparison_note"] = f"could_not_load_mamamia_image:{exc}"
    else:
        out["comparison_note"] = "missing_paired_mamamia_phase"

    recon_pred_mask: np.ndarray | None = None
    if pred_path.exists():
        try:
            pred_img = load_image(pred_path)
            recon_pred_mask = load_mask(pred_path)
            out["prediction_same_grid_as_reconstructed"] = same_grid(pred_img, recon_img)
            out["reconstructed_pred_voxels"] = int(recon_pred_mask.sum())
            out["reconstructed_pred_volume_ml"] = mask_volume_ml(recon_pred_mask, pred_img)
        except Exception as exc:
            out["comparison_note"] = f"could_not_load_reconstructed_prediction:{exc}"
    else:
        out["comparison_note"] = "missing_reconstructed_prediction"

    original_mask: np.ndarray | None = None
    if original_mask_path:
        try:
            mask_img = load_image(original_mask_path)
            original_mask = load_mask(original_mask_path)
            out["mamamia_mask_voxels"] = int(original_mask.sum())
            out["mamamia_mask_volume_ml"] = mask_volume_ml(original_mask, mask_img)
        except Exception as exc:
            out["comparison_note"] = f"could_not_load_mamamia_mask:{exc}"
    else:
        out["comparison_note"] = f"missing_mamamia_{mask_kind}_mask"

    if original_pred_path.exists() and original_mask is not None:
        try:
            orig_pred_img = load_image(original_pred_path)
            mask_img = load_image(original_mask_path)
            orig_pred_mask = load_mask(original_pred_path)
            if same_grid(orig_pred_img, mask_img):
                out[f"mamamia_prediction_vs_{mask_kind}_dice"] = dice(
                    orig_pred_mask, original_mask
                )
                out["mamamia_prediction_voxels"] = int(orig_pred_mask.sum())
                out["mamamia_prediction_volume_ml"] = mask_volume_ml(
                    orig_pred_mask, orig_pred_img
                )
            else:
                out[f"mamamia_prediction_vs_{mask_kind}_dice"] = ""
                out["comparison_note"] = "mamamia_prediction_not_same_grid_as_mask"
        except Exception as exc:
            out["comparison_note"] = f"could_not_compare_mamamia_prediction:{exc}"

    if (
        recon_pred_mask is not None
        and original_mask is not None
        and original_image_path
        and pred_path.exists()
    ):
        pred_img = load_image(pred_path)
        mask_img = load_image(original_mask_path)
        if same_grid(pred_img, mask_img):
            out[f"reconstructed_prediction_vs_{mask_kind}_direct_dice"] = dice(
                recon_pred_mask, original_mask
            )
        else:
            out[f"reconstructed_prediction_vs_{mask_kind}_direct_dice"] = ""
            if not out["comparison_note"]:
                out["comparison_note"] = "native_grids_differ_visual_side_by_side_only"

    return out


def slice_index(mask: np.ndarray | None, fallback_shape: tuple[int, ...]) -> int:
    if mask is not None and mask.ndim >= 3 and mask.any():
        counts = mask.reshape((-1, mask.shape[2])).sum(axis=0)
        return int(np.argmax(counts))
    if len(fallback_shape) >= 3:
        return int(fallback_shape[2] // 2)
    return 0


def normalized_slice(data: np.ndarray, index: int) -> np.ndarray:
    if data.ndim >= 3:
        slc = np.asarray(data[:, :, index], dtype=float)
    else:
        slc = np.asarray(data, dtype=float)
    finite = np.isfinite(slc)
    if not finite.any():
        return np.zeros_like(slc, dtype=float)
    lo, hi = np.percentile(slc[finite], [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((slc - lo) / (hi - lo), 0, 1)


def mask_slice(mask: np.ndarray | None, index: int) -> np.ndarray | None:
    if mask is None:
        return None
    if mask.ndim >= 3:
        return mask[:, :, index].astype(bool)
    return mask.astype(bool)


def overlay_mask(ax, mask: np.ndarray | None, color: tuple[float, float, float], alpha: float) -> None:
    if mask is None or not mask.any():
        return
    rgba = np.zeros((*mask.shape, 4), dtype=float)
    rgba[..., 0] = color[0]
    rgba[..., 1] = color[1]
    rgba[..., 2] = color[2]
    rgba[..., 3] = mask.astype(float) * alpha
    ax.imshow(np.swapaxes(rgba, 0, 1), origin="lower", interpolation="nearest")


def make_overlay(
    row: pd.Series,
    prediction_root: Path,
    output_dir: Path,
    mask_kind: str,
) -> str:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir.parent / ".mplconfig"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recon_image_path = Path(row["source_image_path"])
    recon_pred_path = prediction_path(row, prediction_root)
    mamamia_image_path = Path(str(row.get("paired_mamamia_image_path", "")))
    mamamia_mask_path = Path(str(row.get(f"{mask_kind}_mask_path", "")))
    mamamia_pred_path = original_prediction_path(row, prediction_root)

    if not mamamia_image_path.exists() or not mamamia_mask_path.exists():
        return ""

    mamamia_img = load_image(mamamia_image_path)
    mamamia_data = np.asanyarray(mamamia_img.dataobj)
    mamamia_mask = load_mask(mamamia_mask_path)
    mamamia_pred = load_mask(mamamia_pred_path) if mamamia_pred_path.exists() else None

    recon_img = load_image(recon_image_path)
    recon_data = np.asanyarray(recon_img.dataobj)
    recon_pred = load_mask(recon_pred_path) if recon_pred_path.exists() else None

    mamamia_z = slice_index(mamamia_mask, mamamia_img.shape)
    recon_z = slice_index(recon_pred, recon_img.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=140)

    axes[0].imshow(
        np.swapaxes(normalized_slice(mamamia_data, mamamia_z), 0, 1),
        cmap="gray",
        origin="lower",
    )
    overlay_mask(axes[0], mask_slice(mamamia_mask, mamamia_z), (0.1, 0.9, 0.2), 0.45)
    overlay_mask(axes[0], mask_slice(mamamia_pred, mamamia_z), (0.1, 0.7, 1.0), 0.35)
    axes[0].set_title(f"MAMA-MIA phase {int(row['phase_index'])} + {mask_kind}")
    axes[0].axis("off")

    axes[1].imshow(
        np.swapaxes(normalized_slice(recon_data, recon_z), 0, 1),
        cmap="gray",
        origin="lower",
    )
    overlay_mask(axes[1], mask_slice(recon_pred, recon_z), (0.1, 0.7, 1.0), 0.45)
    axes[1].set_title(f"Reconstructed {row['exam_date']} + nnU-Net")
    axes[1].axis("off")

    fig.suptitle(str(row["patient_id"]))
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        output_dir
        / f"{row['patient_id']}__{row['exam_date']}__phase{int(row['phase_index']):04d}.png"
    )
    fig.savefig(output_path)
    plt.close(fig)
    return str(output_path)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    df = manifest[manifest["collection"] == "reconstructed"].copy()
    if args.patients:
        df = df[df["patient_id"].astype(str).isin(set(args.patients))].copy()
    if args.anchor_only:
        df = df[df["is_anchor_date"].map(is_true)].copy()
    df = df.sort_values(
        ["patient_id", "is_anchor_date", "exam_date", "phase_index"],
        ascending=[True, False, True, True],
        kind="stable",
    )
    if args.max_cases is not None:
        df = df.head(args.max_cases).copy()

    report_rows = [
        summarize_row(row, args.prediction_root.resolve(), args.mask_kind)
        for _, row in df.iterrows()
    ]
    report_df = pd.DataFrame(report_rows)

    overlay_paths: dict[tuple[str, str, int], str] = {}
    if args.make_overlays and not df.empty:
        overlay_df = df.copy()
        if not args.overlay_all_timepoints:
            overlay_df = overlay_df[overlay_df["is_anchor_date"].map(is_true)].copy()
        overlay_df = overlay_df.head(args.max_overlays)
        overlay_dir = output_root / "overlays"
        for _, row in overlay_df.iterrows():
            path = make_overlay(
                row=row,
                prediction_root=args.prediction_root.resolve(),
                output_dir=overlay_dir,
                mask_kind=args.mask_kind,
            )
            key = (str(row["patient_id"]), str(row["exam_date"]), int(row["phase_index"]))
            overlay_paths[key] = path

    if overlay_paths and not report_df.empty:
        report_df["overlay_path"] = [
            overlay_paths.get(
                (str(row.patient_id), str(row.exam_date), int(row.phase_index)), ""
            )
            for row in report_df.itertuples(index=False)
        ]

    report_path = output_root / "nnunet_reconstructed_vs_mamamia_qc_report.csv"
    report_df.to_csv(report_path, index=False)

    print(f"Reconstructed rows processed: {len(report_df)}")
    if not report_df.empty:
        notes = report_df["comparison_note"].fillna("").replace("", "ok")
        print(notes.value_counts().to_string())
        anchors = report_df["is_anchor_date"].map(is_true).sum()
        print(f"Anchor-date rows: {anchors}")
    print(f"Report: {report_path}")
    if overlay_paths:
        print(f"Overlays written: {sum(bool(path) for path in overlay_paths.values())}")


if __name__ == "__main__":
    main()
