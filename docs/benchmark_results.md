# Benchmark Results Snapshot

Snapshot date: 2026-09-15.

These tables summarize the current first-pass pCR benchmark. All rows use the
same official MAMA-MIA train/test split, frozen foundation-model representations,
patient-level embedding aggregation, and L2 logistic-regression probe protocol.

The complete exported result table is available at:

```text
Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv
```

## Evaluation Protocol

| Field | Value |
|---|---|
| Task | Binary pCR prediction |
| Dataset split | Official MAMA-MIA train/test split |
| Final-fit training patients | 1185 |
| Official test patients | 306 |
| Probe | L2-regularized logistic regression |
| Hyperparameter selection | Validation split from official training data only |
| Primary metrics | AUROC, average precision, balanced accuracy |
| Thresholded metrics | Sensitivity, specificity, precision, F1 |

## Clinical Baseline

| Model | Crop | Input | AUROC | AP | Bal Acc | Sensitivity | Specificity | Precision | F1 | Emb Dim | Test N | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clinical baseline | N/A | Clinical variables | 0.735 | 0.502 | 0.642 | 0.419 | 0.864 | 0.574 | 0.484 | N/A | 306 | `clinical_logreg` |

## Best Image-Only Run Per Model

Rows are selected by highest AUROC within each model's completed image-only
first-pass runs. The table still reports AP and balanced accuracy for that same
selected run.

| Model | Crop | Input | AUROC | AP | Bal Acc | Sensitivity | Specificity | Precision | F1 | Emb Dim | Test N | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Pillar-0 BreastMRI | Whole volume | Phase 0 + phase 2 + last phase | 0.546 | 0.358 | 0.527 | 0.581 | 0.474 | 0.325 | 0.417 | 1152 | 306 | `pillar0_breastmri_whole_phase0_phase2_last_logreg` |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | 0.573 | 0.387 | 0.554 | 0.548 | 0.559 | 0.352 | 0.429 | 384 | 306 | `radiodino_expert_roi_phase2_minus_phase0_logreg` |
| BiomedCLIP | Whole volume | Phase 1 | 0.618 | 0.404 | 0.576 | 0.785 | 0.366 | 0.351 | 0.485 | 512 | 306 | `biomedclip_whole_phase1_logreg` |
| Curia | Expert ROI | Phase 2 - phase 0 | 0.630 | 0.434 | 0.578 | 0.419 | 0.737 | 0.411 | 0.415 | 768 | 306 | `curia_expert_roi_phase2_minus_phase0_logreg` |
| MedSigLIP | Expert ROI | Phase 2 - phase 0 | 0.621 | 0.403 | 0.565 | 0.323 | 0.808 | 0.423 | 0.366 | 1152 | 306 | `medsiglip_expert_roi_phase2_minus_phase0_logreg` |

## Best Image-Plus-Clinical Run Per Model

Rows are selected by highest AUROC within each model's completed image-plus-
clinical first-pass runs. Pillar-0 image-plus-clinical and expert-ROI runs are
still pending in this snapshot.

| Model | Crop | Input | AUROC | AP | Bal Acc | Sensitivity | Specificity | Precision | F1 | Emb Dim | Test N | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | 0.706 | 0.501 | 0.638 | 0.516 | 0.761 | 0.485 | 0.500 | 384 | 306 | `radiodino_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| BiomedCLIP | Whole volume | Phase 1 | 0.739 | 0.553 | 0.605 | 0.355 | 0.854 | 0.516 | 0.420 | 512 | 306 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Curia | Expert ROI | Phase 2 - phase 0 | 0.724 | 0.543 | 0.658 | 0.559 | 0.756 | 0.500 | 0.528 | 768 | 306 | `curia_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| MedSigLIP | Whole volume | Phase 1 | 0.706 | 0.483 | 0.579 | 0.290 | 0.869 | 0.491 | 0.365 | 1152 | 306 | `medsiglip_whole_phase1_image_clinical_logreg` |

## Overall Metric Winners

| Setting | Metric won | Model | Crop | Input | Winning value | AUROC | AP | Bal Acc | Run |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Image + clinical | AUROC | BiomedCLIP | Whole volume | Phase 1 | 0.739 | 0.739 | 0.553 | 0.605 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Image + clinical | AP | BiomedCLIP | Whole volume | Phase 1 | 0.553 | 0.739 | 0.553 | 0.605 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Image + clinical | Bal Acc | Curia | Expert ROI | Phase 2 - phase 0 | 0.658 | 0.724 | 0.543 | 0.658 | `curia_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| Image-only | AUROC | Curia | Expert ROI | Phase 2 - phase 0 | 0.630 | 0.630 | 0.434 | 0.578 | `curia_expert_roi_phase2_minus_phase0_logreg` |
| Image-only | AP | Curia | Expert ROI | Selected ROI fusion | 0.446 | 0.608 | 0.446 | 0.546 | `curia_expert_roi_selected_fusion_logreg` |
| Image-only | Bal Acc | Curia | Expert ROI | Phase 2 - phase 0 | 0.578 | 0.630 | 0.434 | 0.578 | `curia_expert_roi_phase2_minus_phase0_logreg` |

## Interpretation

Clinical-only remains a strong reference baseline. The strongest image-only
results currently come from expert-ROI Curia and MedSigLIP runs, suggesting that
tumor-focused representations are more informative than most whole-volume
slice-aggregation settings for pCR. Adding clinical variables improves several
image-based probes, with BiomedCLIP whole-volume phase 1 giving the best
multimodal AUROC/AP and Curia expert-ROI phase2-minus-phase0 giving the best
multimodal balanced accuracy.

The current scientific message is therefore cautious: image embeddings contain
pCR signal, ROI-focused inputs appear helpful, and multimodal fusion can improve
selected metrics, but the clinical-only baseline remains difficult to dominate
across AUROC, AP, and balanced accuracy simultaneously.
