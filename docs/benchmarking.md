# Foundation-Model Benchmarking

The current benchmark evaluates whether frozen medical foundation-model
representations contain useful signal for pCR prediction, and whether they add
predictive value beyond tabular clinical variables.

## Current Benchmark Dataset

The current benchmark uses the original MAMA-MIA official train/test split and
the pretreatment DCE-MRI exam. Longitudinal reconstructed timepoints are the next
dataset extension, but the first-pass benchmark intentionally keeps the task
fixed and comparable.

Target:

- Binary pCR (`pcr`)

Prediction settings:

- clinical-only baseline;
- image-only frozen foundation-model embeddings;
- image-plus-clinical fusion.

Probe protocol:

- frozen encoder;
- patient-level embedding aggregation;
- train-only standardization/imputation;
- L2-regularized logistic regression;
- hyperparameter selection inside official training data only;
- final evaluation on the official MAMA-MIA test set.

Primary metrics:

- AUROC;
- average precision;
- balanced accuracy.

## Model Registry

The full live model table is stored in:

```text
Benchmarking/foundation_model_registry.csv
```

Models currently tracked:

| Priority | Model | Status/role |
|---:|---|---|
| 1 | Pillar-0 BreastMRI | 3D breast MRI foundation-model baseline |
| 2 | MOME Breast mpMRI | Candidate if usable pretrained weights are available |
| 3 | Curia | 2D radiology slice encoder |
| 4 | MedSigLIP | 2D medical image-text encoder |
| 5 | MedImageInsight | Candidate if endpoint/local access is available |
| 6 | BiomedCLIP | Biomedical image-text control |
| 7 | RadImageNet | Radiology CNN transfer baseline |
| 8 | RadioDINO | Self-supervised radiology ViT baseline |
| 9 | RadFM | Heavier 2D/3D radiology VLM candidate |
| 10 | Jolia | Optional CT-oriented cross-modality control |

## Current Results Snapshot

Snapshot date: 2026-09-15.

| Setting | Model | Crop/input | AUROC | AP | Bal Acc | Run |
|---|---|---|---:|---:|---:|---|
| Clinical-only | Clinical baseline | Clinical variables | 0.735 | 0.502 | 0.642 | `clinical_logreg` |
| Best image-only AUROC | Curia | Expert ROI, phase 2 - phase 0 | 0.630 | 0.434 | 0.578 | `curia_expert_roi_phase2_minus_phase0_logreg` |
| Best image-only AP | Curia | Expert ROI, selected ROI fusion | 0.608 | 0.446 | 0.546 | `curia_expert_roi_selected_fusion_logreg` |
| Best image + clinical AUROC/AP | BiomedCLIP | Whole volume, phase 1 + clinical | 0.739 | 0.553 | 0.605 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Best image + clinical balanced accuracy | Curia | Expert ROI, phase 2 - phase 0 + clinical | 0.724 | 0.543 | 0.658 | `curia_expert_roi_phase2_minus_phase0_image_clinical_logreg` |

The key interpretation so far is that clinical-only remains a strong baseline.
The best image-plus-clinical runs improve AP and/or balanced accuracy, but none
clearly dominates clinical-only across all metrics yet.

For detailed per-model tables with sensitivity, specificity, precision, F1,
embedding dimensionality, test-set size, and exact run names, see
[`docs/benchmark_results.md`](benchmark_results.md).

## Result Files

Headline summaries:

- `docs/benchmark_results.md`
- `Benchmarking/outputs/summaries/cross_model_first_pass_summary.md`
- `Benchmarking/outputs/summaries/cross_model_all_results.md`
- `Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv`
- `Benchmarking/outputs/summaries/cross_model_best_by_metric.csv`

Per-model summaries:

- `Benchmarking/outputs/summaries/radiodino_first_pass_summary.md`
- `Benchmarking/outputs/summaries/biomedclip_first_pass_summary.md`
- `Benchmarking/outputs/summaries/curia_first_pass_summary.md`
- `Benchmarking/outputs/summaries/medsiglip_first_pass_summary.md`
- `Benchmarking/outputs/summaries/pillar0_whole_volume_summary.md`

## Next Steps

- Complete Pillar-0 expert-ROI and image-plus-clinical checks.
- Add repeated seeds or cross-validation for shortlisted configurations.
- Extend the benchmark from original MAMA-MIA single-timepoint data to the
  reconstructed longitudinal release.
- Evaluate whether longitudinal features improve pCR prediction beyond
  pretreatment-only imaging and clinical variables.
