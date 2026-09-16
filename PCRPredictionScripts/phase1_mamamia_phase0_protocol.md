# Phase 1: Original MAMA-MIA Phase-0 Image-Only pCR

This starts the pCR pipeline from scratch on the original MAMA-MIA NIfTI files,
using only phase 0 (`*_0000.nii.gz`) and no clinical/tabular features.

## Fair Comparison

The paired models use the same:

- original MAMA-MIA phase-0 whole images
- official MAMA-MIA train/test split
- deterministic validation split from the training patients
- nnU-Net encoder architecture
- classifier head, loss, optimizer, input tensor size, seed, and epochs

The only planned difference is encoder initialization:

- `scratch`: same nnU-Net encoder architecture with randomized encoder weights
- `pretrained`: MAMA-MIA segmentation-pretrained nnU-Net encoder weights

An additional architecture baseline uses `--architecture swin3d_t`, a 3D Swin
Transformer classifier from `torchvision`. It is trained from scratch on the same
MRI tensors and split, so it tests whether a modern 3D classification backbone
behaves differently from the nnU-Net encoder family.

For the cleanest comparison, fine-tune both end-to-end. Freezing the pretrained
encoder is useful as an ablation, but it is not the fairest scratch-vs-pretrained
comparison because the scratch model cannot learn comparable image features.

Note: the released MAMA-MIA nnU-Net weights were trained for segmentation on
MAMA-MIA data. Treat this first experiment as a domain-pretraining comparison.
For a stricter no-overlap claim, later train fold-specific segmentation
pretraining using only training-fold masks.

## Preprocessing

No resized images are written to disk. The training dataset performs:

- NIfTI load
- full-volume crop
- 0.5/99.5 percentile clipping
- per-volume z-score normalization
- aspect-preserving trilinear resize to fit `--input-shape`
- zero padding around the resized volume

Use `--input-shape 256 256 64 --resize-mode pad` for the first real run: 256
in-plane as requested, fixed depth 64 for batching, and padding so the source
volume is not cropped or axis-stretched.

## Weights & Biases Tracking

Training runs log to a separate W&B project:

```text
mamamia-phase1-pcr
```

The trainer logs run configuration, split counts, train/validation metrics each
epoch, best-validation summaries, and final test metrics from the
validation-selected checkpoint. By default the automatic final test evaluation
loads `best.pt`, not the last epoch.
If the machine is not logged in to W&B yet, run `wandb login` once before the
full training commands.

## Build Manifest

```bash
./.venv/bin/python PCRPredictionScripts/build_phase1_mamamia_phase0_manifest.py \
  --output-root PCRPredictionScripts/outputs/phase1_mamamia_phase0
```

## Smoke Checks

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_smoke \
  --image-only \
  --dry-run \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --resize-mode pad \
  --input-shape 256 256 64
```

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t_smoke \
  --architecture swin3d_t \
  --image-only \
  --dry-run \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --resize-mode pad \
  --input-shape 64 64 32
```

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_smoke \
  --image-only \
  --random-init-encoder \
  --dry-run \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --resize-mode pad \
  --input-shape 256 256 64
```

## Train

Run these locally; they are intentionally not launched by Codex because they are
full training jobs.

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_finetune \
  --image-only \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 50 \
  --seed 2026 \
  --resize-mode pad \
  --input-shape 256 256 64 \
  --wandb \
  --wandb-project mamamia-phase1-pcr \
  --wandb-run-name phase1_pretrained_nnunet_finetune \
  --wandb-tags phase1 mamamia phase0 image-only nnunet pretrained
```

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_finetune \
  --image-only \
  --random-init-encoder \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 50 \
  --seed 2026 \
  --resize-mode pad \
  --input-shape 256 256 64 \
  --wandb \
  --wandb-project mamamia-phase1-pcr \
  --wandb-run-name phase1_scratch_nnunet \
  --wandb-tags phase1 mamamia phase0 image-only nnunet scratch
```

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --output-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t \
  --architecture swin3d_t \
  --image-only \
  --device cuda \
  --amp \
  --batch-size 1 \
  --num-workers 2 \
  --epochs 50 \
  --seed 2026 \
  --resize-mode pad \
  --input-shape 256 256 64 \
  --wandb \
  --wandb-project mamamia-phase1-pcr \
  --wandb-run-name phase1_swin3d_t \
  --wandb-tags phase1 mamamia phase0 image-only swin3d transformer
```

If Swin3D exceeds GPU memory at the matched size, rerun it with
`--input-shape 224 224 32` and document that it used a reduced tensor size.

## Evaluate

Use the validation-selected checkpoint for the held-out test set. These commands
write separate `best_on_test` files so the checkpoint choice is explicit.

```bash
./.venv/bin/python PCRPredictionScripts/evaluate_pcr_checkpoint.py \
  --run-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_finetune \
  --checkpoint best.pt \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --split test \
  --output-json PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_finetune/best_on_test.json \
  --output-predictions PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_finetune/best_on_test_predictions.csv
```

```bash
./.venv/bin/python PCRPredictionScripts/evaluate_pcr_checkpoint.py \
  --run-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_finetune \
  --checkpoint best.pt \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --split test \
  --output-json PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_finetune/best_on_test.json \
  --output-predictions PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_finetune/best_on_test_predictions.csv
```

```bash
./.venv/bin/python PCRPredictionScripts/evaluate_pcr_checkpoint.py \
  --run-dir PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t \
  --checkpoint best.pt \
  --manifest PCRPredictionScripts/outputs/phase1_mamamia_phase0/manifest.csv \
  --roi-status mamamia_phase0_whole_image \
  --split test \
  --output-json PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t/best_on_test.json \
  --output-predictions PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t/best_on_test_predictions.csv
```

Then summarize:

```bash
./.venv/bin/python PCRPredictionScripts/summarize_pcr_runs.py \
  PCRPredictionScripts/outputs/phase1_mamamia_phase0_pretrained_finetune \
  PCRPredictionScripts/outputs/phase1_mamamia_phase0_scratch_finetune \
  PCRPredictionScripts/outputs/phase1_mamamia_phase0_swin3d_t \
  --output-csv PCRPredictionScripts/outputs/phase1_mamamia_phase0_comparison.csv
```
