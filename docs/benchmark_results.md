# Foundation-Model Benchmark Results

Snapshot date: 2026-09-22.

The feasible first-pass foundation-model screen is complete. It contains 106
probe runs: 1 clinical-only, 77 image-only, and 28 image-plus-clinical. Of
these, 98 are primary modality-matched/reference runs and 8 are the separate
Jolia CT-to-MRI stress test.

## Evaluation Protocol

| Field | Value |
|---|---|
| Task | Binary pathological complete response prediction |
| Split | Official MAMA-MIA train/test split |
| Final-fit training patients | 1,185 |
| Test patients | 306, including 93 pCR cases |
| Encoders | Frozen foundation models |
| Probe | L2-regularized logistic regression |
| Hyperparameter selection | Inner validation split from official training patients only |
| Threshold selection | Validation balanced accuracy |
| Metrics | AUROC, average precision, balanced accuracy |

Within each run, imputation, scaling, probe hyperparameters, and the decision
threshold are fit without using official test patients. However, many model,
input, crop, and fusion configurations were compared on the same test set.
Headline winners and their bootstrap intervals are therefore exploratory and
test-informed, not confirmatory evidence.

## Clinical Reference

| AUROC | Average precision | Balanced accuracy |
|---:|---:|---:|
| 0.735 | 0.502 | 0.642 |

The clinical-only probe is the reference for deciding whether frozen image
features add useful information.

## Best Image-Only Run Per Primary Model

Rows are selected by AUROC within each model. AP and balanced accuracy are from
the same selected run.

| Model | Crop/input | AUROC | AP | Bal Acc |
|---|---|---:|---:|---:|
| Pillar-0 BreastMRI | Expert ROI, phase 0 + phase 2 + last | 0.586 | 0.403 | 0.563 |
| RadioDINO | Expert ROI, phase 2 - phase 0 | 0.573 | 0.387 | 0.554 |
| BiomedCLIP | Whole volume, phase 1 | 0.618 | 0.404 | 0.576 |
| Curia | Expert ROI, phase 2 - phase 0 | **0.630** | **0.434** | 0.578 |
| MedSigLIP | Expert ROI, phase 2 - phase 0 | 0.621 | 0.403 | 0.565 |
| RadImageNet | Whole volume, phase 2 - phase 0 | 0.619 | 0.396 | **0.593** |

Across all image-only runs, Curia selected expert-ROI fusion has the highest AP
(0.446), although its AUROC is lower at 0.608. No image-only result exceeds the
clinical baseline.

## Best Image-Plus-Clinical Run Per Primary Model

Rows are selected by AUROC within each model.

| Model | Crop/input | AUROC | AP | Bal Acc |
|---|---|---:|---:|---:|
| Pillar-0 BreastMRI | Expert ROI, phase 0 + phase 1 + last | 0.641 | 0.481 | 0.592 |
| RadioDINO | Expert ROI, phase 2 - phase 0 | 0.706 | 0.501 | 0.638 |
| BiomedCLIP | Whole volume, phase 1 | **0.739** | **0.553** | 0.605 |
| Curia | Expert ROI, phase 2 - phase 0 | 0.724 | 0.543 | 0.658 |
| MedSigLIP | Whole volume, phase 1 | 0.706 | 0.483 | 0.579 |
| RadImageNet | Whole volume, phase 2 - phase 0 | 0.725 | 0.510 | **0.681** |

BiomedCLIP has the largest multimodal AUROC and AP point estimates, while
RadImageNet has the largest balanced-accuracy point estimate.

## Paired Bootstrap Against Clinical-Only

The shortlist analysis uses 5,000 paired bootstrap resamples, stratified jointly
by source cohort and pCR class. Positive values favor image plus clinical.

The clinical-only intervals are AUROC 0.735 [0.681, 0.786], AP 0.502
[0.438, 0.599], and balanced accuracy 0.642 [0.591, 0.694].

| Model | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---:|---:|---:|
| BiomedCLIP | +0.004 [-0.037, 0.045] | +0.050 [-0.030, 0.111] | -0.037 [-0.092, 0.017] |
| Curia | -0.011 [-0.070, 0.051] | +0.041 [-0.059, 0.137] | +0.016 [-0.049, 0.082] |
| MedSigLIP | -0.030 [-0.078, 0.020] | -0.019 [-0.109, 0.059] | -0.062 [-0.117, -0.008] |
| Pillar-0 BreastMRI | -0.094 [-0.168, -0.023] | -0.022 [-0.133, 0.076] | -0.049 [-0.118, 0.017] |
| RadImageNet | -0.010 [-0.070, 0.050] | +0.008 [-0.077, 0.094] | +0.039 [-0.022, 0.100] |
| RadioDINO | -0.029 [-0.079, 0.022] | -0.002 [-0.093, 0.079] | -0.003 [-0.057, 0.050] |

No image-plus-clinical configuration has a paired 95% interval showing a stable
aggregate improvement over clinical-only on any primary metric. Pillar-0 is
lower in AUROC, and MedSigLIP is lower in balanced accuracy. These intervals do
not correct for post-selection across the full experiment matrix.

## Source-Cohort Robustness

This analysis divides the official test set by source cohort. It is not an
external generalization test because every source cohort also contributed
patients to the official training split.

| Cohort | Test N | pCR | Clinical AUROC | Best image + clinical AUROC | Main result |
|---|---:|---:|---:|---:|---|
| DUKE | 91 | 30 | 0.787 | 0.754, RadioDINO | No stable AUROC/AP improvement |
| ISPY1 | 67 | 14 | 0.760 | 0.783, BiomedCLIP | Paired AUROC interval crosses zero |
| ISPY2 | 131 | 46 | 0.687 | 0.786, BiomedCLIP | Delta AUROC +0.099 [0.026, 0.177] |
| NACT | 17 | 3 | 0.643 | 0.786, RadImageNet | Too few positive cases for ranking |

In ISPY2, BiomedCLIP plus clinical also improves AP by +0.188 [0.073, 0.267].
This is the clearest cohort-specific hypothesis produced by the benchmark. It
must be retested with source-held-out training or an untouched external cohort
before being interpreted as generalization.

## Jolia Cross-Modality Stress Test

Jolia was trained on adult chest and abdominal CT, so it is reported separately
from the primary breast-MRI leaderboard.

| Setting | Best input | AUROC | AP | Bal Acc |
|---|---|---:|---:|---:|
| Image-only | Whole selected phase fusion | 0.551 | 0.347 | 0.555 |
| Image + clinical | Expert ROI, phase 2 - phase 0 | 0.672 | 0.462 | 0.615 |
| Clinical-only reference | Clinical variables | 0.735 | 0.502 | 0.642 |

The result is consistent with the expected CT-to-MRI domain mismatch. Jolia is
a useful negative transfer control, not a competitive MRI model.

## Registry Exclusions

- **MOME Breast mpMRI:** the released method requires a multiparametric input
  set including DWI and T2, which is not available in the MAMA-MIA benchmark.
- **MedImageInsight:** access is through an Azure Limited Preview rather than a
  locally downloadable checkpoint suitable for this workflow.
- **RadFM:** the official full-model path requires an NVIDIA A100 80 GB-class
  setup, a roughly 50 GB checkpoint, and a legacy software stack; no supported
  standalone visual encoder is released for the current hardware path.

## Interpretation

The completed screen shows that frozen foundation embeddings contain modest pCR
signal, but it does not establish aggregate added value beyond clinical data.
ROI cropping helps Curia and MedSigLIP, while BiomedCLIP and RadImageNet favor
whole-volume inputs. More feature concatenation is not consistently better than
a well-chosen phase or subtraction.

The most defensible next experiment is a predeclared leave-one-source-dataset-
out comparison of clinical-only and a small frozen shortlist, with BiomedCLIP
as the primary candidate. A supervised CNN/ViT baseline should then establish
whether frozen embeddings are preferable to task-specific learning before the
project moves to longitudinal modeling.

## Reproducible Outputs

- `Benchmarking/outputs/summaries/probe_run_summary.csv`
- `Benchmarking/outputs/summaries/cross_model_all_results.md`
- `Benchmarking/outputs/summaries/cross_model_best_by_metric.csv`
- `Benchmarking/outputs/summaries/bootstrap_shortlist_summary.md`
- `Benchmarking/outputs/summaries/bootstrap_shortlist_intervals.csv`
- `Benchmarking/outputs/summaries/bootstrap_shortlist_deltas_vs_clinical.csv`
- `Benchmarking/outputs/summaries/shortlist_by_dataset_summary.md`
- `Benchmarking/outputs/summaries/shortlist_by_dataset_intervals.csv`
- `Benchmarking/outputs/summaries/shortlist_by_dataset_deltas_vs_clinical.csv`
- `Benchmarking/outputs/summaries/jolia_cross_modality_summary.md`
