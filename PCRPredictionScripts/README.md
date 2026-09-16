# Phase-0 nnU-Net pCR Classifier

This is the first conservative pCR experiment:

1. use the pretrained MAMA-MIA nnU-Net only for tumor localization,
2. make a fixed physical crop around accepted phase-0 predicted masks,
3. initialize a pCR classifier from the same nnU-Net encoder,
4. optionally fuse MRI features with clinical variables from MAMA-MIA.

## Build the ROI manifest

```bash
./.venv/bin/python PCRPredictionScripts/build_phase0_roi_manifest.py \
  --qc-report ReconstructionScripts/qc_reports/mamamia_nnunet_qc_phase0_anchor/nnunet_reconstructed_vs_mamamia_qc_report.csv \
  --output-root PCRPredictionScripts/outputs/phase0_nnunet_roi
```

Default QC gates:

- predicted mask is non-empty
- predicted volume is at least `0.05 ml`
- predicted/MAMA-MIA mask volume ratio is between `0.2` and `5.0`
- crop is centered on all positive predicted-mask voxels
- crop physical size is `96 x 96 x 64 mm`

For slower but cleaner center estimates, add `--component-mode largest`.

## Smoke-test the classifier

This only loads the data, loads the MAMA-MIA nnU-Net encoder weights, and runs one forward pass.

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase0_nnunet_roi/roi_manifest.csv \
  --dry-run \
  --device cpu \
  --batch-size 1 \
  --num-workers 0 \
  --input-shape 64 64 64
```

## Train

Run the actual training locally when the smoke test passes.

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase0_nnunet_roi/roi_manifest.csv \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --epochs 30 \
  --input-shape 96 96 64 \
  --freeze-encoder
```

Remove `--freeze-encoder` after the first baseline if you want to fine-tune the
encoder. Use `--image-only` for an MRI-only baseline without clinical variables.
Use `--random-init-encoder` for the from-scratch control: same nnU-Net encoder
architecture, but with the encoder weights reset instead of using MAMA-MIA
segmentation pretraining.

Recommended methodology after the first two multimodal runs:

1. Train MRI-only models with `--image-only`.
2. Compare tumor ROI versus whole-image again.
3. Compare frozen pretrained encoder versus fine-tuned pretrained encoder.
4. Train a random-init encoder as a control, not as the expected main model.
5. Add multimodal fusion to the best MRI-only setup.

## Comparison Model: Whole-Image, No Mask QC

This baseline uses all labeled phase-0 anchor images and ignores the segmentation
mask quality. The classifier architecture is the same, but the crop is the whole
reconstructed volume resized to the network input shape.

```bash
./.venv/bin/python PCRPredictionScripts/build_phase0_whole_image_manifest.py \
  --output-root PCRPredictionScripts/outputs/phase0_whole_image
```

```bash
./.venv/bin/python PCRPredictionScripts/train_phase0_nnunet_pcr_classifier.py \
  --manifest PCRPredictionScripts/outputs/phase0_whole_image/roi_manifest.csv \
  --roi-status whole_image \
  --output-dir PCRPredictionScripts/outputs/phase0_whole_image_pcr_classifier \
  --device cuda \
  --batch-size 1 \
  --num-workers 0 \
  --epochs 30 \
  --input-shape 96 96 64 \
  --freeze-encoder
```

After the comparison model finishes, summarize both runs:

```bash
./.venv/bin/python PCRPredictionScripts/summarize_pcr_runs.py \
  PCRPredictionScripts/outputs/phase0_nnunet_pcr_classifier \
  PCRPredictionScripts/outputs/phase0_whole_image_pcr_classifier \
  --output-csv PCRPredictionScripts/outputs/phase0_model_comparison.csv
```

For a fairer test comparison, evaluate both `best.pt` checkpoints on the shared
test patients that have accepted nnU-Net tumor crops:

```bash
./.venv/bin/python PCRPredictionScripts/evaluate_pcr_checkpoint.py \
  --run-dir PCRPredictionScripts/outputs/phase0_nnunet_pcr_classifier \
  --checkpoint best.pt \
  --manifest PCRPredictionScripts/outputs/phase0_nnunet_roi/roi_manifest.csv \
  --roi-status nnunet_pass \
  --split test \
  --output-json PCRPredictionScripts/outputs/phase0_nnunet_pcr_classifier/best_on_roi_test.json \
  --output-predictions PCRPredictionScripts/outputs/phase0_nnunet_pcr_classifier/best_on_roi_test_predictions.csv
```

```bash
./.venv/bin/python PCRPredictionScripts/evaluate_pcr_checkpoint.py \
  --run-dir PCRPredictionScripts/outputs/phase0_whole_image_pcr_classifier \
  --checkpoint best.pt \
  --manifest PCRPredictionScripts/outputs/phase0_whole_image/roi_manifest.csv \
  --roi-status whole_image \
  --split test \
  --patient-ids-from PCRPredictionScripts/outputs/phase0_nnunet_roi/roi_manifest.csv \
  --patient-ids-from-roi-status nnunet_pass \
  --patient-ids-from-split test \
  --output-json PCRPredictionScripts/outputs/phase0_whole_image_pcr_classifier/best_on_roi_test.json \
  --output-predictions PCRPredictionScripts/outputs/phase0_whole_image_pcr_classifier/best_on_roi_test_predictions.csv
```
