# Foundation Model pCR Benchmark Matrix

This file is the working experiment checklist for the foundation-model pCR
benchmark. The goal is to keep the benchmark interpretable:

1. Clinical-only baseline: how well non-image variables predict pCR.
2. Image-only foundation embeddings: whether the frozen image representation
   contains pCR signal.
3. Image plus clinical fusion: whether image embeddings add signal beyond
   clinical variables.

## Fixed Benchmark Rules

- Dataset: MAMA-MIA official train/test split from
  `Benchmarking/outputs/manifests/mamamia_multiphase_foundation_manifest.csv`.
- Target: binary pCR.
- Foundation model status: frozen encoder.
- Probe: L2-regularized logistic regression, not a deep network.
- Hyperparameter selection: inner validation split from official train only.
- Final evaluation: official MAMA-MIA test only.
- Primary metrics: AUROC, average precision, balanced accuracy.

Within each run, model and threshold selection use training data only. However,
the cross-model shortlist and headline winners were chosen after comparing many
configurations on the same official test set. Winner rankings and bootstrap
intervals are therefore exploratory rather than confirmatory.

## Current Status

Snapshot date: 2026-09-22. The feasible registry screen is complete with 106
probe runs: 1 clinical-only, 77 image-only, and 28 image-plus-clinical. Of
these, 98 are primary modality-matched/reference runs and 8 are the separate
Jolia CT-to-MRI stress test.

| Priority | Model/Input | Crop | Feature Set | Clinical Data | Status | Test AUROC | Test AP | Test Bal Acc | Output |
|---:|---|---|---|---|---|---:|---:|---:|---|
| 0 | Clinical variables | N/A | Clinical-only | Yes | Done | 0.735 | 0.502 | 0.642 | `Benchmarking/outputs/probes/clinical_logreg` |
| 1 | RadioDINO phase0 | Whole volume | Image-only | No | Done | 0.537 | 0.343 | 0.515 | `Benchmarking/outputs/probes/radiodino_whole_phase0_logreg` |
| 1 | RadioDINO phase1 | Whole volume | Image-only | No | Done | 0.547 | 0.364 | 0.528 | `Benchmarking/outputs/probes/radiodino_whole_phase1_logreg` |
| 1 | RadioDINO phase2 | Whole volume | Image-only | No | Done | 0.489 | 0.296 | 0.488 | `Benchmarking/outputs/probes/radiodino_whole_phase2_logreg` |
| 1 | RadioDINO last phase | Whole volume | Image-only | No | Done | 0.467 | 0.284 | 0.502 | `Benchmarking/outputs/probes/radiodino_whole_last_phase_logreg` |
| 1 | RadioDINO phase1 - phase0 | Whole volume | Image-only | No | Done | 0.528 | 0.353 | 0.512 | `Benchmarking/outputs/probes/radiodino_whole_phase1_minus_phase0_logreg` |
| 1 | RadioDINO phase2 - phase0 | Whole volume | Image-only | No | Done | 0.549 | 0.327 | 0.553 | `Benchmarking/outputs/probes/radiodino_whole_phase2_minus_phase0_logreg` |
| 1 | RadioDINO last phase - phase0 | Whole volume | Image-only | No | Done | 0.530 | 0.322 | 0.544 | `Benchmarking/outputs/probes/radiodino_whole_last_phase_minus_phase0_logreg` |
| 2 | RadioDINO raw phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.523 | 0.342 | 0.522 | `Benchmarking/outputs/probes/radiodino_whole_raw_phases_fusion_logreg` |
| 2 | RadioDINO subtraction fusion | Whole volume | Image-only fused embeddings | No | Done | 0.526 | 0.354 | 0.513 | `Benchmarking/outputs/probes/radiodino_whole_subtractions_fusion_logreg` |
| 2 | RadioDINO all DCE fusion | Whole volume | Image-only fused embeddings | No | Done | 0.548 | 0.351 | 0.542 | `Benchmarking/outputs/probes/radiodino_whole_all_dce_fusion_logreg` |
| 3 | RadioDINO phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.640 | 0.407 | 0.592 | `Benchmarking/outputs/probes/radiodino_whole_phase1_image_clinical_logreg` |
| 3 | RadioDINO phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.701 | 0.475 | 0.622 | `Benchmarking/outputs/probes/radiodino_whole_phase2_minus_phase0_image_clinical_logreg` |
| 3 | RadioDINO all DCE fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.614 | 0.405 | 0.602 | `Benchmarking/outputs/probes/radiodino_whole_all_dce_fusion_image_clinical_logreg` |
| 4 | RadioDINO phase1 | Expert ROI | Image-only | No | Done | 0.546 | 0.376 | 0.522 | `Benchmarking/outputs/probes/radiodino_expert_roi_phase1_logreg` |
| 4 | RadioDINO phase2 - phase0 | Expert ROI | Image-only | No | Done | 0.573 | 0.387 | 0.554 | `Benchmarking/outputs/probes/radiodino_expert_roi_phase2_minus_phase0_logreg` |
| 5 | RadioDINO selected ROI fusion | Expert ROI | Image-only fused embeddings | No | Done | 0.556 | 0.364 | 0.536 | `Benchmarking/outputs/probes/radiodino_expert_roi_selected_fusion_logreg` |
| 5 | RadioDINO phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.706 | 0.501 | 0.638 | `Benchmarking/outputs/probes/radiodino_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 6 | BiomedCLIP phase1 | Whole volume | Image-only | No | Done | 0.618 | 0.404 | 0.576 | `Benchmarking/outputs/probes/biomedclip_whole_phase1_logreg` |
| 6 | BiomedCLIP phase2 - phase0 | Whole volume | Image-only | No | Done | 0.542 | 0.341 | 0.519 | `Benchmarking/outputs/probes/biomedclip_whole_phase2_minus_phase0_logreg` |
| 6 | BiomedCLIP phase2 - phase0 | Expert ROI | Image-only | No | Done | 0.549 | 0.363 | 0.549 | `Benchmarking/outputs/probes/biomedclip_expert_roi_phase2_minus_phase0_logreg` |
| 7 | BiomedCLIP phase0 | Whole volume | Image-only | No | Done | 0.572 | 0.383 | 0.535 | `Benchmarking/outputs/probes/biomedclip_whole_phase0_logreg` |
| 7 | BiomedCLIP phase2 | Whole volume | Image-only | No | Done | 0.584 | 0.397 | 0.536 | `Benchmarking/outputs/probes/biomedclip_whole_phase2_logreg` |
| 7 | BiomedCLIP last phase | Whole volume | Image-only | No | Done | 0.591 | 0.381 | 0.567 | `Benchmarking/outputs/probes/biomedclip_whole_last_phase_logreg` |
| 7 | BiomedCLIP phase1 - phase0 | Whole volume | Image-only | No | Done | 0.604 | 0.382 | 0.529 | `Benchmarking/outputs/probes/biomedclip_whole_phase1_minus_phase0_logreg` |
| 7 | BiomedCLIP last phase - phase0 | Whole volume | Image-only | No | Done | 0.546 | 0.352 | 0.504 | `Benchmarking/outputs/probes/biomedclip_whole_last_phase_minus_phase0_logreg` |
| 8 | BiomedCLIP raw phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.583 | 0.371 | 0.560 | `Benchmarking/outputs/probes/biomedclip_whole_raw_phases_fusion_logreg` |
| 8 | BiomedCLIP subtraction fusion | Whole volume | Image-only fused embeddings | No | Done | 0.556 | 0.364 | 0.530 | `Benchmarking/outputs/probes/biomedclip_whole_subtractions_fusion_logreg` |
| 8 | BiomedCLIP all DCE fusion | Whole volume | Image-only fused embeddings | No | Done | 0.602 | 0.409 | 0.544 | `Benchmarking/outputs/probes/biomedclip_whole_all_dce_fusion_logreg` |
| 9 | BiomedCLIP phase1 | Expert ROI | Image-only | No | Done | 0.580 | 0.380 | 0.561 | `Benchmarking/outputs/probes/biomedclip_expert_roi_phase1_logreg` |
| 9 | BiomedCLIP phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.739 | 0.553 | 0.605 | `Benchmarking/outputs/probes/biomedclip_whole_phase1_image_clinical_logreg` |
| 9 | BiomedCLIP phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.686 | 0.452 | 0.613 | `Benchmarking/outputs/probes/biomedclip_whole_phase2_minus_phase0_image_clinical_logreg` |
| 9 | BiomedCLIP all DCE fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.670 | 0.468 | 0.624 | `Benchmarking/outputs/probes/biomedclip_whole_all_dce_fusion_image_clinical_logreg` |
| 9 | BiomedCLIP selected ROI fusion | Expert ROI | Image-only fused embeddings | No | Done | 0.565 | 0.361 | 0.531 | `Benchmarking/outputs/probes/biomedclip_expert_roi_selected_fusion_logreg` |
| 9 | BiomedCLIP phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.706 | 0.513 | 0.652 | `Benchmarking/outputs/probes/biomedclip_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 10 | Curia phase0 | Whole volume | Image-only | No | Done | 0.480 | 0.306 | 0.497 | `Benchmarking/outputs/probes/curia_whole_phase0_logreg` |
| 10 | Curia phase1 | Whole volume | Image-only | No | Done | 0.501 | 0.306 | 0.487 | `Benchmarking/outputs/probes/curia_whole_phase1_logreg` |
| 10 | Curia phase2 | Whole volume | Image-only | No | Done | 0.526 | 0.350 | 0.501 | `Benchmarking/outputs/probes/curia_whole_phase2_logreg` |
| 10 | Curia last phase | Whole volume | Image-only | No | Done | 0.527 | 0.338 | 0.489 | `Benchmarking/outputs/probes/curia_whole_last_phase_logreg` |
| 10 | Curia phase1 - phase0 | Whole volume | Image-only | No | Done | 0.473 | 0.287 | 0.498 | `Benchmarking/outputs/probes/curia_whole_phase1_minus_phase0_logreg` |
| 10 | Curia phase2 - phase0 | Whole volume | Image-only | No | Done | 0.513 | 0.316 | 0.522 | `Benchmarking/outputs/probes/curia_whole_phase2_minus_phase0_logreg` |
| 10 | Curia last phase - phase0 | Whole volume | Image-only | No | Done | 0.533 | 0.345 | 0.518 | `Benchmarking/outputs/probes/curia_whole_last_phase_minus_phase0_logreg` |
| 11 | Curia raw phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.543 | 0.341 | 0.513 | `Benchmarking/outputs/probes/curia_whole_raw_phases_fusion_logreg` |
| 11 | Curia subtraction fusion | Whole volume | Image-only fused embeddings | No | Done | 0.483 | 0.306 | 0.463 | `Benchmarking/outputs/probes/curia_whole_subtractions_fusion_logreg` |
| 11 | Curia all DCE fusion | Whole volume | Image-only fused embeddings | No | Done | 0.531 | 0.343 | 0.505 | `Benchmarking/outputs/probes/curia_whole_all_dce_fusion_logreg` |
| 12 | Curia phase1 | Expert ROI | Image-only | No | Done | 0.602 | 0.430 | 0.543 | `Benchmarking/outputs/probes/curia_expert_roi_phase1_logreg` |
| 12 | Curia phase2 - phase0 | Expert ROI | Image-only | No | Done | 0.630 | 0.434 | 0.578 | `Benchmarking/outputs/probes/curia_expert_roi_phase2_minus_phase0_logreg` |
| 12 | Curia selected ROI fusion | Expert ROI | Image-only fused embeddings | No | Done | 0.608 | 0.446 | 0.546 | `Benchmarking/outputs/probes/curia_expert_roi_selected_fusion_logreg` |
| 13 | Curia phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.673 | 0.443 | 0.582 | `Benchmarking/outputs/probes/curia_whole_phase1_image_clinical_logreg` |
| 13 | Curia phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.653 | 0.445 | 0.588 | `Benchmarking/outputs/probes/curia_whole_phase2_minus_phase0_image_clinical_logreg` |
| 13 | Curia all DCE fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.621 | 0.400 | 0.617 | `Benchmarking/outputs/probes/curia_whole_all_dce_fusion_image_clinical_logreg` |
| 13 | Curia phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.724 | 0.543 | 0.658 | `Benchmarking/outputs/probes/curia_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 10 | Curia full first-pass ladder | Whole volume and expert ROI | Image-only, fusion, image + clinical | Mixed | Done | 0.724 | 0.543 | 0.658 | `Benchmarking/outputs/summaries/curia_first_pass_summary.md` |
| 14 | MedSigLIP smoke test | Whole volume | Image-only | No | Done | N/A | N/A | N/A | `Benchmarking/outputs/logs/medsiglip_cpu_tiny_smoke.log` |
| 14 | MedSigLIP phase0 | Whole volume | Image-only | No | Done | 0.541 | 0.331 | 0.541 | `Benchmarking/outputs/probes/medsiglip_whole_phase0_logreg` |
| 14 | MedSigLIP phase1 | Whole volume | Image-only | No | Done | 0.543 | 0.366 | 0.533 | `Benchmarking/outputs/probes/medsiglip_whole_phase1_logreg` |
| 14 | MedSigLIP phase2 | Whole volume | Image-only | No | Done | 0.590 | 0.389 | 0.542 | `Benchmarking/outputs/probes/medsiglip_whole_phase2_logreg` |
| 14 | MedSigLIP last phase | Whole volume | Image-only | No | Done | 0.593 | 0.378 | 0.546 | `Benchmarking/outputs/probes/medsiglip_whole_last_phase_logreg` |
| 14 | MedSigLIP phase1 - phase0 | Whole volume | Image-only | No | Done | 0.587 | 0.363 | 0.562 | `Benchmarking/outputs/probes/medsiglip_whole_phase1_minus_phase0_logreg` |
| 14 | MedSigLIP phase2 - phase0 | Whole volume | Image-only | No | Done | 0.585 | 0.390 | 0.544 | `Benchmarking/outputs/probes/medsiglip_whole_phase2_minus_phase0_logreg` |
| 14 | MedSigLIP last phase - phase0 | Whole volume | Image-only | No | Done | 0.561 | 0.347 | 0.542 | `Benchmarking/outputs/probes/medsiglip_whole_last_phase_minus_phase0_logreg` |
| 14 | MedSigLIP raw phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.546 | 0.371 | 0.529 | `Benchmarking/outputs/probes/medsiglip_whole_raw_phases_fusion_logreg` |
| 14 | MedSigLIP subtraction fusion | Whole volume | Image-only fused embeddings | No | Done | 0.565 | 0.347 | 0.555 | `Benchmarking/outputs/probes/medsiglip_whole_subtractions_fusion_logreg` |
| 14 | MedSigLIP all DCE fusion | Whole volume | Image-only fused embeddings | No | Done | 0.547 | 0.369 | 0.520 | `Benchmarking/outputs/probes/medsiglip_whole_all_dce_fusion_logreg` |
| 14 | MedSigLIP phase1 | Expert ROI | Image-only | No | Done | 0.576 | 0.379 | 0.551 | `Benchmarking/outputs/probes/medsiglip_expert_roi_phase1_logreg` |
| 14 | MedSigLIP phase2 - phase0 | Expert ROI | Image-only | No | Done | 0.621 | 0.403 | 0.565 | `Benchmarking/outputs/probes/medsiglip_expert_roi_phase2_minus_phase0_logreg` |
| 14 | MedSigLIP selected ROI fusion | Expert ROI | Image-only fused embeddings | No | Done | 0.619 | 0.404 | 0.578 | `Benchmarking/outputs/probes/medsiglip_expert_roi_selected_fusion_logreg` |
| 14 | MedSigLIP phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.706 | 0.483 | 0.579 | `Benchmarking/outputs/probes/medsiglip_whole_phase1_image_clinical_logreg` |
| 14 | MedSigLIP phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.699 | 0.482 | 0.632 | `Benchmarking/outputs/probes/medsiglip_whole_phase2_minus_phase0_image_clinical_logreg` |
| 14 | MedSigLIP all DCE fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.639 | 0.426 | 0.621 | `Benchmarking/outputs/probes/medsiglip_whole_all_dce_fusion_image_clinical_logreg` |
| 14 | MedSigLIP phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.703 | 0.481 | 0.653 | `Benchmarking/outputs/probes/medsiglip_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 14 | MedSigLIP full first-pass ladder | Whole volume and expert ROI | Image-only, fusion, image + clinical | Mixed | Done | 0.706 | 0.483 | 0.653 | `Benchmarking/outputs/summaries/medsiglip_first_pass_summary.md` |
| 15 | Pillar-0 BreastMRI smoke test | Whole-volume 3D DCE | Image-only | No | Done | N/A | N/A | N/A | `Benchmarking/outputs/logs/pillar0_whole_phase0_phase1_last_smoke.log` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + phase2 | Whole-volume 3D DCE | Image-only | No | Done | 0.542 | 0.366 | 0.531 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_phase2_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + last | Whole-volume 3D DCE | Image-only | No | Done | 0.526 | 0.357 | 0.523 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_last_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase2 + last | Whole-volume 3D DCE | Image-only | No | Done | 0.546 | 0.358 | 0.527 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase2_last_logreg` |
| 15 | Pillar-0 BreastMRI subtraction triplet | Whole-volume 3D DCE | Image-only | No | Done | 0.486 | 0.318 | 0.494 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_subtractions_phase1_phase2_last_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + phase2 | Expert ROI 3D DCE | Image-only | No | Done | 0.557 | 0.369 | 0.533 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_phase0_phase1_phase2_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + last | Expert ROI 3D DCE | Image-only | No | Done | 0.572 | 0.392 | 0.527 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_phase0_phase1_last_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase2 + last | Expert ROI 3D DCE | Image-only | No | Done | 0.586 | 0.403 | 0.563 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_phase0_phase2_last_logreg` |
| 15 | Pillar-0 BreastMRI subtraction triplet | Expert ROI 3D DCE | Image-only | No | Done | 0.568 | 0.381 | 0.552 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_subtractions_phase1_phase2_last_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + last + clinical | Whole-volume 3D DCE | Image + clinical | Yes | Done | 0.606 | 0.421 | 0.573 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_last_image_clinical_logreg` |
| 15 | Pillar-0 BreastMRI subtraction triplet + clinical | Whole-volume 3D DCE | Image + clinical | Yes | Done | 0.602 | 0.384 | 0.555 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_subtractions_phase1_phase2_last_image_clinical_logreg` |
| 15 | Pillar-0 BreastMRI phase0 + phase1 + last + clinical | Expert ROI 3D DCE | Image + clinical | Yes | Done | 0.641 | 0.481 | 0.592 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_phase0_phase1_last_image_clinical_logreg` |
| 15 | Pillar-0 BreastMRI subtraction triplet + clinical | Expert ROI 3D DCE | Image + clinical | Yes | Done | 0.631 | 0.457 | 0.585 | `Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_subtractions_phase1_phase2_last_image_clinical_logreg` |
| 15 | Pillar-0 BreastMRI full first-pass ladder | Whole volume and expert ROI | 3D image-only and image + clinical | Mixed | Done | 0.641 | 0.481 | 0.592 | `Benchmarking/outputs/summaries/cross_model_first_pass_summary.md` |
| 16 | RadImageNet phase0 | Whole volume | Image-only | No | Done | 0.528 | 0.346 | 0.530 | `Benchmarking/outputs/probes/radimagenet_whole_phase0_logreg` |
| 16 | RadImageNet phase1 | Whole volume | Image-only | No | Done | 0.528 | 0.335 | 0.518 | `Benchmarking/outputs/probes/radimagenet_whole_phase1_logreg` |
| 16 | RadImageNet phase2 | Whole volume | Image-only | No | Done | 0.600 | 0.396 | 0.522 | `Benchmarking/outputs/probes/radimagenet_whole_phase2_logreg` |
| 16 | RadImageNet last phase | Whole volume | Image-only | No | Done | 0.582 | 0.384 | 0.526 | `Benchmarking/outputs/probes/radimagenet_whole_last_phase_logreg` |
| 16 | RadImageNet phase1 - phase0 | Whole volume | Image-only | No | Done | 0.586 | 0.352 | 0.551 | `Benchmarking/outputs/probes/radimagenet_whole_phase1_minus_phase0_logreg` |
| 16 | RadImageNet phase2 - phase0 | Whole volume | Image-only | No | Done | 0.619 | 0.396 | 0.593 | `Benchmarking/outputs/probes/radimagenet_whole_phase2_minus_phase0_logreg` |
| 16 | RadImageNet last phase - phase0 | Whole volume | Image-only | No | Done | 0.551 | 0.343 | 0.525 | `Benchmarking/outputs/probes/radimagenet_whole_last_phase_minus_phase0_logreg` |
| 16 | RadImageNet raw phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.497 | 0.301 | 0.491 | `Benchmarking/outputs/probes/radimagenet_whole_raw_phases_fusion_logreg` |
| 16 | RadImageNet subtraction fusion | Whole volume | Image-only fused embeddings | No | Done | 0.604 | 0.376 | 0.582 | `Benchmarking/outputs/probes/radimagenet_whole_subtractions_fusion_logreg` |
| 16 | RadImageNet all DCE fusion | Whole volume | Image-only fused embeddings | No | Done | 0.570 | 0.388 | 0.556 | `Benchmarking/outputs/probes/radimagenet_whole_all_dce_fusion_logreg` |
| 16 | RadImageNet phase1 | Expert ROI | Image-only | No | Done | 0.529 | 0.324 | 0.509 | `Benchmarking/outputs/probes/radimagenet_expert_roi_phase1_logreg` |
| 16 | RadImageNet phase2 - phase0 | Expert ROI | Image-only | No | Done | 0.565 | 0.351 | 0.514 | `Benchmarking/outputs/probes/radimagenet_expert_roi_phase2_minus_phase0_logreg` |
| 16 | RadImageNet selected ROI fusion | Expert ROI | Image-only fused embeddings | No | Done | 0.531 | 0.309 | 0.488 | `Benchmarking/outputs/probes/radimagenet_expert_roi_selected_fusion_logreg` |
| 16 | RadImageNet phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.642 | 0.434 | 0.606 | `Benchmarking/outputs/probes/radimagenet_whole_phase1_image_clinical_logreg` |
| 16 | RadImageNet phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.725 | 0.510 | 0.681 | `Benchmarking/outputs/probes/radimagenet_whole_phase2_minus_phase0_image_clinical_logreg` |
| 16 | RadImageNet all DCE fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.603 | 0.414 | 0.571 | `Benchmarking/outputs/probes/radimagenet_whole_all_dce_fusion_image_clinical_logreg` |
| 16 | RadImageNet phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.651 | 0.408 | 0.568 | `Benchmarking/outputs/probes/radimagenet_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 16 | RadImageNet full first-pass ladder | Whole volume and expert ROI | Image-only, fusion, image + clinical | Mixed | Done | 0.725 | 0.510 | 0.681 | `Benchmarking/outputs/summaries/cross_model_first_pass_summary.md` |
| 17 | MedImageInsight compatibility and access audit | N/A | Cloud embedding endpoint | N/A | Access blocked | N/A | N/A | N/A | Azure Limited Preview requires approval, deployment, endpoint calls, and governance review; no local checkpoint is available. |
| 18 | Jolia CT-to-MRI transfer smoke test | Whole-volume phase1, one patient | Image-only | No | Done | N/A | N/A | N/A | 576-D embedding, CUDA, zero failures. |
| 18 | Jolia phase1 transfer | Whole volume | Image-only | No | Done | 0.491 | 0.305 | 0.475 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_phase1_logreg` |
| 18 | Jolia phase2 - phase0 transfer | Whole volume | Image-only | No | Done | 0.489 | 0.321 | 0.515 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_phase2_minus_phase0_logreg` |
| 18 | Jolia phase2 - phase0 transfer | Expert ROI | Image-only | No | Done | 0.539 | 0.335 | 0.522 | `Benchmarking/outputs/probes/jolia_cross_modality_expert_roi_phase2_minus_phase0_logreg` |
| 18 | Jolia selected phase fusion | Whole volume | Image-only fused embeddings | No | Done | 0.551 | 0.347 | 0.555 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_selected_fusion_logreg` |
| 18 | Jolia phase1 + clinical | Whole volume | Image + clinical | Yes | Done | 0.635 | 0.424 | 0.593 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_phase1_image_clinical_logreg` |
| 18 | Jolia phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | Done | 0.669 | 0.431 | 0.535 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_phase2_minus_phase0_image_clinical_logreg` |
| 18 | Jolia phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | Done | 0.672 | 0.462 | 0.615 | `Benchmarking/outputs/probes/jolia_cross_modality_expert_roi_phase2_minus_phase0_image_clinical_logreg` |
| 18 | Jolia selected phase fusion + clinical | Whole volume | Image + clinical | Yes | Done | 0.605 | 0.414 | 0.551 | `Benchmarking/outputs/probes/jolia_cross_modality_whole_selected_fusion_image_clinical_logreg` |
| 18 | Jolia predeclared transfer screen | Phase1 and phase2 - phase0, whole and selected ROI | Image-only, selected fusion, image + clinical | Mixed | Done | 0.672 | 0.462 | 0.615 | `Benchmarking/outputs/summaries/jolia_cross_modality_summary.md` |
| 19 | RadFM resource feasibility audit | N/A | Hardware, disk, and software compatibility | N/A | Resource blocked | N/A | N/A | N/A | `Benchmarking/outputs/logs/radfm_resource_audit.txt` |
| 20 | Shortlist paired bootstrap | Official test set | Clinical-only, image-only, and image + clinical | Mixed | Done | N/A | N/A | N/A | `Benchmarking/outputs/summaries/bootstrap_shortlist_summary.md` |
| 21 | Shortlist source-cohort robustness audit | DUKE, ISPY1, ISPY2, and NACT test subgroups | Clinical-only, image-only, and image + clinical | Mixed | Done | N/A | N/A | N/A | `Benchmarking/outputs/summaries/shortlist_by_dataset_summary.md` |

## Interpretation So Far

The completed benchmark supports four main conclusions:

1. Clinical-only is the strongest aggregate reference (AUROC 0.735, AP 0.502,
   balanced accuracy 0.642). No image-plus-clinical shortlist configuration has
   a paired 95% bootstrap interval showing a stable aggregate improvement over
   it.
2. Frozen image embeddings contain modest pCR signal. Curia expert-ROI
   phase2-minus-phase0 has the best image-only AUROC (0.630), Curia selected ROI
   fusion has the best image-only AP (0.446), and RadImageNet whole-volume
   phase2-minus-phase0 has the best image-only balanced accuracy (0.593).
3. The best point estimates after clinical fusion are BiomedCLIP whole-volume
   phase1 for AUROC/AP (0.739/0.553) and RadImageNet whole-volume
   phase2-minus-phase0 for balanced accuracy (0.681), but their paired aggregate
   improvements remain uncertain.
4. Source-cohort behavior is heterogeneous. BiomedCLIP plus clinical shows the
   only convincing cohort-specific incremental signal in ISPY2: delta AUROC
   +0.099 [0.026, 0.177] and delta AP +0.188 [0.073, 0.267]. This remains an
   exploratory subgroup result, not external validation, because ISPY2 also
   contributed training patients.

Whole-volume RadioDINO image-only performance is close to chance. The best
whole-volume image-only AUROC is about 0.55, which is much lower than the
clinical-only baseline. ROI cropping gives a modest improvement for the
phase2-minus-phase0 subtraction input: AUROC rises from 0.549 to 0.573 and AP
rises from 0.327 to 0.387. ROI phase1 is roughly similar to whole-volume phase1.
ROI fusion of phase1 plus phase2-minus-phase0 does not help; it is worse than
the single ROI subtraction input. ROI plus clinical is the best RadioDINO
multimodal result so far, but it still does not clearly beat clinical-only:
AUROC is lower than clinical-only, AP is essentially tied, and balanced accuracy
is slightly lower.

BiomedCLIP is stronger than RadioDINO for image-only pCR prediction. Its best
single image-only result is whole-volume phase1, with AUROC 0.618, AP 0.404, and
balanced accuracy 0.576. Whole-volume all-DCE fusion has the best BiomedCLIP
image-only AP at 0.409, but does not improve AUROC over phase1 alone. ROI does
not help BiomedCLIP as much as it helped RadioDINO, and selected ROI fusion is
weaker than the best single whole-volume phase. The best overall multimodal
result so far is BiomedCLIP whole phase1 plus clinical, with AUROC 0.739 and AP
0.553. BiomedCLIP expert-ROI phase2-minus-phase0 plus clinical has the best
selected-threshold balanced accuracy at 0.652.

Curia first-pass benchmarking is complete. Whole-volume pCR signal is weak: raw
phase fusion is the best whole-volume AUROC at 0.543, while phase2 is the best
whole-volume AP at 0.350. All-DCE fusion does not improve this. Expert ROI
cropping changes the Curia result substantially: expert-ROI phase2-minus-phase0
is the best Curia image-only result by AUROC, with AUROC 0.630, AP 0.434, and
balanced accuracy 0.578. Selected ROI fusion has the best Curia image-only AP
at 0.446 but lower AUROC. The best Curia multimodal result is expert-ROI
phase2-minus-phase0 plus clinical, with AUROC 0.724, AP 0.543, and balanced
accuracy 0.658. This is slightly below BiomedCLIP phase1 plus clinical by
AUROC/AP, but it has the strongest selected-threshold balanced accuracy so far.

MedSigLIP whole-volume benchmarking is complete. The best single whole-volume
AUROC is last phase at 0.593, while the best whole-volume AP is
phase2-minus-phase0 at 0.390. Fusion does not improve over the best single
MedSigLIP whole-volume inputs: raw phase fusion reaches AUROC 0.546, subtraction
fusion reaches AUROC 0.565, and all-DCE fusion reaches AUROC 0.547. The
following completed stage was expert ROI extraction, motivated by the ROI gains
seen for Curia and RadioDINO.

MedSigLIP expert ROI image-only benchmarking is also complete. ROI improves the
best MedSigLIP image-only AUROC from 0.593 to 0.621 using expert-ROI
phase2-minus-phase0. Selected ROI fusion has very similar AUROC at 0.619 and the
best MedSigLIP image-only AP/balanced accuracy at 0.404/0.578. The standardized
image-plus-clinical probes were then completed.

MedSigLIP first-pass benchmarking is complete. The best MedSigLIP multimodal
AUROC/AP comes from whole phase1 plus clinical at 0.706/0.483. The best
MedSigLIP multimodal balanced accuracy comes from expert-ROI
phase2-minus-phase0 plus clinical at 0.653. MedSigLIP improves substantially
when clinical variables are added, but it does not beat the current best
BiomedCLIP AUROC/AP or Curia balanced accuracy multimodal results.

Pillar-0 BreastMRI was benchmarked as a 3D breast MRI foundation model rather
than another 2D slice encoder. Its first-pass grid uses
3-channel DCE volumes: phase0/phase1/phase2, phase0/phase1/last, phase0/phase2/last,
and subtraction triplets. This is a model-specific benchmark that answers
whether a native 3D breast MRI foundation representation improves pCR prediction
over the completed 2D slice-aggregation baselines.

Pillar-0 whole-volume image-only benchmarking is complete. All four 3-channel
3D DCE inputs embedded 1491 patients with zero extraction failures and produced
1152-dimensional representations. The whole-volume results are weak: the best
AUROC is 0.546 for phase0/phase2/last, the best AP is 0.366 for
phase0/phase1/phase2, and the subtraction triplet is below chance by AUROC. This
suggests that whole-volume Pillar-0 embeddings do not currently outperform the
stronger 2D foundation-model embeddings or the clinical-only baseline for pCR
prediction. Expert ROI was therefore evaluated as the corresponding localization
check because several previous models improved after focusing on the tumor.

Pillar-0 first-pass benchmarking is complete. Its best image-only result is
expert-ROI phase0/phase2/last (AUROC 0.586, AP 0.403, balanced accuracy 0.563),
and its best image-plus-clinical result is expert-ROI phase0/phase1/last
(0.641/0.481/0.592). These remain below the clinical-only baseline.

RadImageNet first-pass benchmarking is complete. Its strongest image-only input
is whole-volume phase2-minus-phase0 (AUROC 0.619, AP 0.396, balanced accuracy
0.593). ROI cropping and feature concatenation both reduce performance. Adding
clinical variables to the whole-volume phase2 subtraction reaches AUROC 0.725,
AP 0.510, and balanced accuracy 0.681. This is the current best image-only and
image-plus-clinical balanced accuracy, while BiomedCLIP remains best for
multimodal AUROC and AP.

MedImageInsight Premium was also audited, but the current release is an Azure
Limited Preview rather than a locally downloadable checkpoint. It is recorded
as access-blocked pending approval, deployment, cost, and data-governance
decisions. Jolia was executed after its Hugging Face terms were accepted.
Because Jolia was trained only on adult chest/abdominal CT and its
model card warns against transfer to other modalities, its MRI results will be
reported as a separate cross-modality stress test rather than ranked beside the
modality-matched models.

Jolia whole-volume phase1 extraction is complete for all 1,491 eligible
patients with zero failures and 576-dimensional global embeddings. Its
image-only pCR result is at chance: AUROC 0.491, AP 0.305 (approximately equal
to the 0.304 test-set pCR prevalence), and balanced accuracy 0.475. The
validation-selected threshold of 0.998 and poor test log loss indicate unstable
transfer under the MRI-to-CT input adapter. The predeclared
phase2-minus-phase0 screen was subsequently completed before drawing the final
transfer conclusion.

Jolia whole-volume phase2-minus-phase0 is also complete for all 1,491 patients
with zero extraction failures. It remains at chance by AUROC (0.489), with AP
0.321 and balanced accuracy 0.515. Because both whole-volume inputs use the
same extreme validation-selected threshold of 0.998, the remaining expert-ROI
subtraction run is treated as a final localization check, not evidence that the
CT representation is already transferring to breast MRI.

Jolia expert-ROI phase2-minus-phase0 is the strongest of the three image-only
stress tests, but remains weak: AUROC 0.539, AP 0.335, and balanced accuracy
0.522. Its validation-selected threshold (0.464) and test log loss (0.682) are
substantially more stable than the whole-volume runs. This supports the narrow
interpretation that localization reduces domain-mismatch noise, not that a CT
foundation model is competitive with the modality-matched MRI encoders.

Jolia's complete predeclared stress test confirms weak CT-to-MRI transfer.
Whole phase fusion is its best image-only configuration (AUROC 0.551, AP 0.347,
balanced accuracy 0.555). Expert-ROI phase2-minus-phase0 plus clinical is its
best multimodal configuration (0.672/0.462/0.615), but it remains below the
clinical-only baseline (0.735/0.502/0.642) on all three metrics. Jolia is
therefore retained as a negative cross-modality control rather than promoted
to the primary breast-MRI model shortlist.

RadFM's resource audit is complete. The available accelerator is an AMD Radeon
PRO W7900 with 45 GiB VRAM under ROCm, with 30 GiB system RAM and approximately
460 GiB free disk. This does not satisfy the official full-model path, which
documents an NVIDIA A100 80 GB requirement and ships a roughly 50 GB checkpoint
using a legacy Transformers stack. The release has no standalone visual
checkpoint or supported vision-only loader. RadFM is therefore retained in the
registry as a transparent resource exclusion; no weights were downloaded.

The 5,000-resample paired bootstrap is complete for the 14-run exploratory
shortlist. None of the image-plus-clinical configurations has a paired 95%
interval showing a stable improvement over clinical-only. BiomedCLIP's AUROC
difference is +0.004 [-0.037, 0.045], Curia's AP difference is +0.041
[-0.059, 0.137], and RadImageNet's balanced-accuracy difference is +0.039
[-0.022, 0.100]. MedSigLIP is lower than clinical-only in balanced accuracy,
and Pillar-0 is lower in AUROC. These are descriptive post-selection intervals,
not confirmatory significance tests.

The complete result table is written to
`Benchmarking/outputs/summaries/cross_model_all_results.md`. This table should
be treated as the appendix/source table: it includes every completed run, while
the cross-model first-pass summary keeps the headline standardized comparisons.

## Benchmark Closeout And Follow-Up

The feasible foundation-model screen, aggregate paired bootstrap, and source-
cohort robustness audit are complete. No additional foundation-model extraction
is required for this benchmark stage. The completed result tables can be
regenerated with:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
```

The next experiment should be predeclared before touching the official test
set again. The strongest candidate is a leave-one-source-dataset-out analysis
of the clinical baseline and a small frozen shortlist, especially BiomedCLIP,
followed by a separately evaluated supervised CNN/ViT baseline and then the
longitudinal extension.

## Saved Results

RadioDINO first-pass results have been saved in:

```text
Benchmarking/outputs/summaries/radiodino_first_pass_summary.md
Benchmarking/outputs/summaries/radiodino_first_pass_probe_summary.csv
```

BiomedCLIP first-pass results have been saved in:

```text
Benchmarking/outputs/summaries/biomedclip_first_pass_summary.md
Benchmarking/outputs/summaries/biomedclip_first_pass_probe_summary.csv
```

Curia first-pass results have been saved in:

```text
Benchmarking/outputs/summaries/curia_first_pass_summary.md
Benchmarking/outputs/summaries/curia_first_pass_probe_summary.csv
```

MedSigLIP first-pass results have been saved in:

```text
Benchmarking/outputs/summaries/medsiglip_first_pass_summary.md
Benchmarking/outputs/summaries/medsiglip_first_pass_probe_summary.csv
```

Cross-model first-pass comparison results have been saved in:

```text
Benchmarking/outputs/summaries/cross_model_first_pass_summary.md
Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv
Benchmarking/outputs/summaries/cross_model_all_results.md
Benchmarking/outputs/summaries/cross_model_best_by_metric.csv
Benchmarking/outputs/summaries/cross_model_best_image_only.png
Benchmarking/outputs/summaries/cross_model_best_image_plus_clinical.png
```

Uncertainty, cohort, and transfer-control results are saved in:

```text
Benchmarking/outputs/summaries/bootstrap_shortlist_summary.md
Benchmarking/outputs/summaries/bootstrap_shortlist_intervals.csv
Benchmarking/outputs/summaries/bootstrap_shortlist_deltas_vs_clinical.csv
Benchmarking/outputs/summaries/shortlist_by_dataset_summary.md
Benchmarking/outputs/summaries/shortlist_by_dataset_intervals.csv
Benchmarking/outputs/summaries/shortlist_by_dataset_deltas_vs_clinical.csv
Benchmarking/outputs/summaries/jolia_cross_modality_summary.md
Benchmarking/outputs/summaries/jolia_cross_modality_probe_summary.csv
```
