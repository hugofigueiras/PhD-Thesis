# RadImageNet Benchmark Commands

RadImageNet ResNet-50 is the next runnable benchmark after Pillar-0. It accepts
2D MRI slices directly and therefore uses the same patient split, 16-slice
sampling, patient-level mean aggregation, expert ROI, fusion, and logistic
probe protocol as the other 2D foundation models.

MOME was audited first but excluded from the primary DCE benchmark because its
released architecture requires six DCE subtraction channels, DWI, and T2.
MAMA-MIA does not contain the required DWI/T2 inputs.

## 1. Download And Inspect The Official Weights

```bash
mkdir -p Benchmarking/outputs/model_weights/radimagenet

./.venv/bin/gdown \
  'https://drive.google.com/uc?id=1RHt2GnuOYlc_gcoTETtBDSW73mFyRAtR' \
  -O Benchmarking/outputs/model_weights/radimagenet/RadImageNet_pytorch.zip

unzip -l Benchmarking/outputs/model_weights/radimagenet/RadImageNet_pytorch.zip
```

Extract the archive. The default ResNet-50 state-dict location is:

```text
Benchmarking/outputs/model_weights/radimagenet/extracted/RadImageNet_pytorch/ResNet50.pt
```

The extractor loads this checkpoint with `strict=True`; a mismatched or
partially loaded checkpoint is treated as an error.

## 2. One-Patient CUDA Smoke Test

```bash
mkdir -p Benchmarking/outputs/logs

./.venv/bin/python -u Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key radimagenet \
  --input-role phase1 \
  --crop-mode whole \
  --max-patients 1 \
  --max-slices 1 \
  --batch-size 1 \
  --device auto \
  --log-every 1 \
  > Benchmarking/outputs/logs/radimagenet_whole_phase1_smoke.log 2>&1

status=$?
tail -80 Benchmarking/outputs/logs/radimagenet_whole_phase1_smoke.log
printf '\nRadImageNet smoke exit status: %s\n' "$status"
```

Do not start the full ladder until the smoke test exits with status 0 and the
summary reports a 2048-dimensional embedding.
