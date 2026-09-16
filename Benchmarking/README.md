# Foundation Model Benchmarking

This folder contains the first benchmark layer for the advisor-requested
foundation-model study.

The immediate task is binary pCR prediction on the original MAMA-MIA
pretreatment DCE-MRI exam. The benchmark starts with frozen foundation-model
embeddings and lightweight probes before any model fine-tuning.

## Starting Point

Build the multiphase benchmark manifest:

```bash
./.venv/bin/python Benchmarking/build_mamamia_multiphase_manifest.py
```

The full run reads NIfTI headers and expert masks once to audit geometry and ROI
crop coordinates. It can take a few minutes, but it does not create image crops,
subtraction images, embeddings, or training tensors.

The script writes:

```text
Benchmarking/outputs/manifests/mamamia_multiphase_foundation_manifest.csv
Benchmarking/outputs/manifests/mamamia_multiphase_foundation_summary.json
```

The manifest has one row per MAMA-MIA patient and records:

- pCR label and official split
- phase 0, phase 1, phase 2, and last-phase image paths
- subtraction availability for post-contrast minus phase 0
- expert and automatic mask paths
- expert ROI crop coordinates with a fixed physical margin
- flags for whole-volume, subtraction/multiphase, and expert-ROI benchmark use
- clinical variables, with a leakage-safe clinical feature policy in the summary

## Model Registry

The model list lives in:

```text
Benchmarking/foundation_model_registry.csv
```

It includes the foundation models from the methodology document:

- Pillar-0 BreastMRI
- MOME Breast mpMRI
- Curia
- MedSigLIP
- MedImageInsight
- BiomedCLIP
- RadImageNet
- RadioDINO
- RadFM
- Jolia

## Benchmark Ladder

The live experiment checklist lives in:

```text
Benchmarking/experiment_matrix.md
```

Run this order once embedding extraction scripts exist:

1. clinical-only probe
2. whole-volume phase-0 image probe
3. whole-volume single post-contrast phase probes
4. whole-volume subtraction probes
5. whole-volume multiphase fusion
6. expert-ROI single phase and subtraction probes
7. expert-ROI multiphase fusion
8. whole-volume plus ROI fusion
9. image plus clinical fusion

Do not start with full fine-tuning or reconstructed longitudinal timepoints. Those
belong after the baseline foundation-model embedding benchmark is stable.

## Probe Architecture

All first-pass probes use:

```text
frozen representation or clinical variables
  -> train-only standardization / imputation
  -> L2-regularized logistic regression
  -> pCR probability
```

The official MAMA-MIA test split is never used to select hyperparameters. Each
probe creates a stratified validation split only from official training patients,
selects `C` by validation average precision, chooses a classification threshold
by validation balanced accuracy, then fits the final probe on all official
training patients before evaluating the official test patients.

Run the clinical-only baseline:

```bash
./.venv/bin/python Benchmarking/train_clinical_probe.py
```

The script writes:

```text
Benchmarking/outputs/probes/clinical_logreg/
  metrics.json
  test_predictions.csv
  train_predictions.csv
  coefficients.csv
  model.joblib
  run_config.json
```

After one or more probe runs, summarize them:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py \
  Benchmarking/outputs/probes/clinical_logreg
```

## Embedding Probe Contract

Extract a small RadioDINO smoke set first:

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key radiodino \
  --input-role phase0 \
  --crop-mode whole \
  --max-patients 8 \
  --max-slices 8 \
  --device cpu
```

This writes patient-level embeddings to:

```text
Benchmarking/outputs/embeddings/radiodino/whole_phase0/patient_embeddings.csv
```

Once a foundation-model extractor writes patient-level embeddings for enough
train and test patients, train the same logistic probe with:

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/<model>/<input>/patient_embeddings.csv \
  --feature-prefix emb_ \
  --output-dir Benchmarking/outputs/probes/<model>_<input>_logreg
```

The embedding CSV must contain one row per patient:

```text
patient_id,emb_0000,emb_0001,emb_0002,...
```

Use `--feature-prefix emb_` if the file includes additional metadata columns.
