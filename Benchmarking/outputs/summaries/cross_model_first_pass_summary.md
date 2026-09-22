# Cross-Model First-Pass Benchmark Summary

Snapshot date: 2026-09-22

This compares the completed first-pass foundation-model benchmarks using
the same MAMA-MIA official train/test split, frozen encoders, patient-level
embedding aggregation, and L2 logistic regression probe protocol.

Full tables:

```text
Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv
Benchmarking/outputs/summaries/cross_model_all_results.md
Benchmarking/outputs/summaries/cross_model_best_by_metric.csv
Benchmarking/outputs/summaries/cross_model_best_image_only.png
Benchmarking/outputs/summaries/cross_model_best_image_plus_clinical.png
```

## Clinical Baseline

| model | AUROC | AP | Bal Acc |
|---|---:|---:|---:|
| Clinical baseline | 0.735 | 0.502 | 0.642 |

## Best Image-Only Run Per Model

| model | crop | input | AUROC | AP | Bal Acc |
|---|---|---|---:|---:|---:|
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 2 + last phase | 0.586 | 0.403 | 0.563 |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | 0.573 | 0.387 | 0.554 |
| BiomedCLIP | Whole volume | Phase 1 | 0.618 | 0.404 | 0.576 |
| Curia | Expert ROI | Phase 2 - phase 0 | 0.630 | 0.434 | 0.578 |
| MedSigLIP | Expert ROI | Phase 2 - phase 0 | 0.621 | 0.403 | 0.565 |
| RadImageNet | Whole volume | Phase 2 - phase 0 | 0.619 | 0.396 | 0.593 |

## Best Image-Plus-Clinical Run Per Model

| model | crop | input | AUROC | AP | Bal Acc |
|---|---|---|---:|---:|---:|
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | 0.641 | 0.481 | 0.592 |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | 0.706 | 0.501 | 0.638 |
| BiomedCLIP | Whole volume | Phase 1 | 0.739 | 0.553 | 0.605 |
| Curia | Expert ROI | Phase 2 - phase 0 | 0.724 | 0.543 | 0.658 |
| MedSigLIP | Whole volume | Phase 1 | 0.706 | 0.483 | 0.579 |
| RadImageNet | Whole volume | Phase 2 - phase 0 | 0.725 | 0.510 | 0.681 |

## Overall Metric Winners

| criterion | model | crop | input | Best Value | AUROC | AP | Bal Acc |
|---|---|---|---|---:|---:|---:|---:|
| Image + clinical best AUROC | BiomedCLIP | Whole volume | Phase 1 | 0.739 | 0.739 | 0.553 | 0.605 |
| Image + clinical best AP | BiomedCLIP | Whole volume | Phase 1 | 0.553 | 0.739 | 0.553 | 0.605 |
| Image + clinical best Bal Acc | RadImageNet | Whole volume | Phase 2 - phase 0 | 0.681 | 0.725 | 0.510 | 0.681 |
| Image-only best AUROC | Curia | Expert ROI | Phase 2 - phase 0 | 0.630 | 0.630 | 0.434 | 0.578 |
| Image-only best AP | Curia | Expert ROI | Selected ROI fusion | 0.446 | 0.608 | 0.446 | 0.546 |
| Image-only best Bal Acc | RadImageNet | Whole volume | Phase 2 - phase 0 | 0.593 | 0.619 | 0.396 | 0.593 |

## Interpretation

The best image-only AUROC is 0.630 from Curia expert-ROI phase2-minus-phase0. The best image-only AP is 0.446 from Curia expert-ROI selected ROI fusion, and the best image-only balanced accuracy is 0.593 from RadImageNet whole-volume phase2-minus-phase0.

For multimodal prediction, the best AUROC is 0.739 from BiomedCLIP whole-volume phase1. The best AP is 0.553 from BiomedCLIP whole-volume phase1, and the best selected-threshold balanced accuracy is 0.681 from RadImageNet whole-volume phase2-minus-phase0.

Clinical-only remains a very strong baseline. The best image-plus-clinical
runs improve AP and/or balanced accuracy, but none of them clearly
dominates clinical-only on all three metrics. That is the key scientific
message for this first-pass benchmark.

## Recommended Shortlist

Keep these configurations for deeper validation and later repeated-seed or
cross-validation checks:

- BiomedCLIP whole-volume phase1: best multimodal AUROC.
- BiomedCLIP whole-volume phase1: best multimodal AP.
- RadImageNet whole-volume phase2-minus-phase0: best multimodal balanced accuracy.
- Curia expert-ROI phase2-minus-phase0: best image-only AUROC.
- Clinical-only: mandatory reference baseline.
