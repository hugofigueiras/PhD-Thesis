# Foundation Model Benchmarking

This folder contains the frozen-foundation-model benchmark for pathological
complete response (pCR) prediction from the original MAMA-MIA pretreatment
DCE-MRI exam and clinical variables.

## Status

Snapshot date: 2026-09-22.

- Feasible registry screen complete.
- 106 completed probes: 1 clinical-only, 77 image-only, and 28
  image-plus-clinical.
- 98 primary modality-matched/reference runs and 8 separate Jolia CT-to-MRI
  stress-test runs.
- Aggregate 5,000-resample paired bootstrap complete.
- DUKE, ISPY1, ISPY2, and NACT source-cohort robustness audit complete.

The [illustrated benchmark report](../docs/benchmark_results.md) summarizes the
design, results, uncertainty analysis, and scientific interpretation. The live
row-level checklist is in [`experiment_matrix.md`](experiment_matrix.md), and
the reproducible result tables are in [`outputs/summaries/`](outputs/summaries/).

## Benchmark Overview

![Foundation-model benchmarking pipeline](../presentations/assets/benchmarking_pipeline.svg)

The frozen image branch and training-fit clinical branch produce three directly
comparable feature settings: clinical-only, image-only, and image plus clinical.
Only the L2-regularized logistic-regression probe is trained for the downstream
pCR task.

## Fixed Protocol

| Item | Choice |
|---|---|
| Target | Binary pCR |
| Split | Official MAMA-MIA train/test split |
| Final-fit train/test N | 1,185 / 306 |
| Encoder | Frozen |
| Probe | L2-regularized logistic regression |
| Selection | Inner validation split from official training patients |
| Metrics | AUROC, average precision, balanced accuracy |

Within each run, preprocessing, regularization, and threshold selection use
training data only. Across runs, many configurations were compared on the same
official test set, so leaderboard winners and bootstrap intervals are
exploratory and test-informed.

## Experiment Settings

1. Clinical-only establishes the non-image reference.
2. Image-only tests whether a frozen representation contains pCR signal.
3. Image plus clinical tests whether the image representation adds information
   beyond clinical variables.

Inputs include whole-volume and expert-ROI crops; raw DCE phases;
post-contrast-minus-phase0 subtractions; and selected feature-level fusions.
Pillar-0 receives 3D phase triplets as channels before its encoder. The 2D
models aggregate sampled slice features and concatenate patient embeddings for
phase fusion.

![DCE-MRI image-fusion strategies](../presentations/assets/fusion_strategy.svg)

Expert-ROI inputs use a tumor bounding box derived from the phase-0 MAMA-MIA
expert mask, expanded by a 30 x 30 x 20 mm physical margin. The same clipped
coordinates are applied to every phase and subtraction input.

![Expert-ROI crop strategy](../presentations/assets/roi_crop_strategy.svg)

## Completed Primary Models

- Pillar-0 BreastMRI
- RadioDINO
- BiomedCLIP
- Curia
- MedSigLIP
- RadImageNet

Jolia is reported separately as a CT-to-MRI negative transfer control. MOME is
excluded because the required DWI/T2 inputs are unavailable, MedImageInsight is
access-blocked behind an Azure Limited Preview, and RadFM is resource-blocked
on the current supported hardware/software path.

## Headline Results

| Setting | Model/input | AUROC | AP | Bal Acc |
|---|---|---:|---:|---:|
| Clinical-only | Clinical variables | 0.735 | 0.502 | 0.642 |
| Best image-only AUROC | Curia ROI subtraction | 0.630 | 0.434 | 0.578 |
| Best image-only AP | Curia ROI selected fusion | 0.608 | 0.446 | 0.546 |
| Best image-only Bal Acc | RadImageNet whole subtraction | 0.619 | 0.396 | 0.593 |
| Best multimodal AUROC/AP | BiomedCLIP whole phase 1 + clinical | 0.739 | 0.553 | 0.605 |
| Best multimodal Bal Acc | RadImageNet whole subtraction + clinical | 0.725 | 0.510 | 0.681 |

No primary image-plus-clinical shortlist run has a paired 95% bootstrap interval
showing a stable aggregate improvement over clinical-only. In the ISPY2 test
subgroup, BiomedCLIP plus clinical has exploratory improvements of +0.099
[0.026, 0.177] AUROC and +0.188 [0.073, 0.267] AP.

### Image-Only Comparison

![Best image-only run per primary model](../presentations/assets/benchmark_best_image_only.png)

### Image-Plus-Clinical Comparison

![Best image-plus-clinical run per primary model](../presentations/assets/benchmark_best_image_plus_clinical.png)

## Core Workflow

Build the manifest:

```bash
./.venv/bin/python Benchmarking/build_mamamia_multiphase_manifest.py
```

Train the clinical baseline:

```bash
./.venv/bin/python Benchmarking/train_clinical_probe.py
```

Extract a small 2D smoke set:

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key radiodino \
  --input-role phase0 \
  --crop-mode whole \
  --max-patients 8 \
  --max-slices 8 \
  --device cpu
```

Train a probe from an embedding table:

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/<model>/<input>/patient_embeddings.csv \
  --feature-prefix emb_ \
  --output-dir Benchmarking/outputs/probes/<model>_<input>_logreg
```

Regenerate completed summaries without rerunning image extraction:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
./.venv/bin/python Benchmarking/bootstrap_benchmark_shortlist.py --n-bootstrap 5000 --seed 20260921
./.venv/bin/python Benchmarking/analyze_shortlist_by_dataset.py --n-bootstrap 5000 --seed 20260922
```

## Key Outputs

- `outputs/manifests/mamamia_multiphase_foundation_manifest.csv`
- `outputs/summaries/probe_run_summary.csv`
- `outputs/summaries/cross_model_all_results.md`
- `outputs/summaries/cross_model_best_by_metric.csv`
- `outputs/summaries/bootstrap_shortlist_summary.md`
- `outputs/summaries/shortlist_by_dataset_summary.md`
- `outputs/summaries/jolia_cross_modality_summary.md`

The next stage should lock the current official test set and predeclare a
source-generalization experiment, followed by a supervised CNN/ViT baseline and
then longitudinal modeling.
