# Cross-Model All Probe Results

Snapshot date: 2026-09-15

This table intentionally includes every completed probe run, not only the
headline standardized subset. Use it as the appendix/source table for the
foundation-model benchmark.

## Clinical-only

| model | crop | input | feature_set | Clinical | AUROC | AP | Bal Acc | official_test | Run |
|---|---|---|---|---|---:|---:|---:|---|---|
| Clinical baseline | N/A | Clinical variables | Clinical-only | Yes | 0.735 | 0.502 | 0.642 | 306 | clinical_logreg |

## Image-only

| model | crop | input | feature_set | Clinical | AUROC | AP | Bal Acc | official_test | Run |
|---|---|---|---|---|---:|---:|---:|---|---|
| Pillar-0 BreastMRI | Whole volume | Phase 0 + phase 1 + last phase | Image-only | No | 0.526 | 0.357 | 0.523 | 306 | pillar0_breastmri_whole_phase0_phase1_last_logreg |
| Pillar-0 BreastMRI | Whole volume | Phase 0 + phase 1 + phase 2 | Image-only | No | 0.542 | 0.366 | 0.531 | 306 | pillar0_breastmri_whole_phase0_phase1_phase2_logreg |
| Pillar-0 BreastMRI | Whole volume | Phase 0 + phase 2 + last phase | Image-only | No | 0.546 | 0.358 | 0.527 | 306 | pillar0_breastmri_whole_phase0_phase2_last_logreg |
| Pillar-0 BreastMRI | Whole volume | Subtraction triplet | Image-only | No | 0.486 | 0.318 | 0.494 | 306 | pillar0_breastmri_whole_subtractions_phase1_phase2_last_logreg |
| RadioDINO | Whole volume | All DCE fusion | Image-only fused embeddings | No | 0.548 | 0.351 | 0.542 | 306 | radiodino_whole_all_dce_fusion_logreg |
| RadioDINO | Whole volume | Last phase | Image-only | No | 0.467 | 0.284 | 0.502 | 306 | radiodino_whole_last_phase_logreg |
| RadioDINO | Whole volume | Last phase - phase 0 | Image-only | No | 0.530 | 0.322 | 0.544 | 306 | radiodino_whole_last_phase_minus_phase0_logreg |
| RadioDINO | Whole volume | Phase 0 | Image-only | No | 0.537 | 0.343 | 0.515 | 306 | radiodino_whole_phase0_logreg |
| RadioDINO | Whole volume | Phase 1 | Image-only | No | 0.547 | 0.364 | 0.528 | 306 | radiodino_whole_phase1_logreg |
| RadioDINO | Whole volume | Phase 1 - phase 0 | Image-only | No | 0.528 | 0.353 | 0.512 | 306 | radiodino_whole_phase1_minus_phase0_logreg |
| RadioDINO | Whole volume | Phase 2 | Image-only | No | 0.489 | 0.296 | 0.488 | 306 | radiodino_whole_phase2_logreg |
| RadioDINO | Whole volume | Phase 2 - phase 0 | Image-only | No | 0.549 | 0.327 | 0.553 | 306 | radiodino_whole_phase2_minus_phase0_logreg |
| RadioDINO | Whole volume | Raw phase fusion | Image-only fused embeddings | No | 0.523 | 0.342 | 0.522 | 306 | radiodino_whole_raw_phases_fusion_logreg |
| RadioDINO | Whole volume | Subtraction fusion | Image-only fused embeddings | No | 0.526 | 0.354 | 0.513 | 306 | radiodino_whole_subtractions_fusion_logreg |
| RadioDINO | Expert ROI | Phase 1 | Image-only | No | 0.546 | 0.376 | 0.522 | 306 | radiodino_expert_roi_phase1_logreg |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | Image-only | No | 0.573 | 0.387 | 0.554 | 306 | radiodino_expert_roi_phase2_minus_phase0_logreg |
| RadioDINO | Expert ROI | Selected ROI fusion | Image-only fused embeddings | No | 0.556 | 0.364 | 0.536 | 306 | radiodino_expert_roi_selected_fusion_logreg |
| BiomedCLIP | Whole volume | All DCE fusion | Image-only fused embeddings | No | 0.602 | 0.409 | 0.544 | 306 | biomedclip_whole_all_dce_fusion_logreg |
| BiomedCLIP | Whole volume | Last phase | Image-only | No | 0.591 | 0.381 | 0.567 | 306 | biomedclip_whole_last_phase_logreg |
| BiomedCLIP | Whole volume | Last phase - phase 0 | Image-only | No | 0.546 | 0.352 | 0.504 | 306 | biomedclip_whole_last_phase_minus_phase0_logreg |
| BiomedCLIP | Whole volume | Phase 0 | Image-only | No | 0.572 | 0.383 | 0.535 | 306 | biomedclip_whole_phase0_logreg |
| BiomedCLIP | Whole volume | Phase 1 | Image-only | No | 0.618 | 0.404 | 0.576 | 306 | biomedclip_whole_phase1_logreg |
| BiomedCLIP | Whole volume | Phase 1 - phase 0 | Image-only | No | 0.604 | 0.382 | 0.529 | 306 | biomedclip_whole_phase1_minus_phase0_logreg |
| BiomedCLIP | Whole volume | Phase 2 | Image-only | No | 0.584 | 0.397 | 0.536 | 306 | biomedclip_whole_phase2_logreg |
| BiomedCLIP | Whole volume | Phase 2 - phase 0 | Image-only | No | 0.542 | 0.341 | 0.519 | 306 | biomedclip_whole_phase2_minus_phase0_logreg |
| BiomedCLIP | Whole volume | Raw phase fusion | Image-only fused embeddings | No | 0.583 | 0.371 | 0.560 | 306 | biomedclip_whole_raw_phases_fusion_logreg |
| BiomedCLIP | Whole volume | Subtraction fusion | Image-only fused embeddings | No | 0.556 | 0.364 | 0.530 | 306 | biomedclip_whole_subtractions_fusion_logreg |
| BiomedCLIP | Expert ROI | Phase 1 | Image-only | No | 0.580 | 0.380 | 0.561 | 306 | biomedclip_expert_roi_phase1_logreg |
| BiomedCLIP | Expert ROI | Phase 2 - phase 0 | Image-only | No | 0.549 | 0.363 | 0.549 | 306 | biomedclip_expert_roi_phase2_minus_phase0_logreg |
| BiomedCLIP | Expert ROI | Selected ROI fusion | Image-only fused embeddings | No | 0.565 | 0.361 | 0.531 | 306 | biomedclip_expert_roi_selected_fusion_logreg |
| Curia | Whole volume | All DCE fusion | Image-only fused embeddings | No | 0.531 | 0.343 | 0.505 | 306 | curia_whole_all_dce_fusion_logreg |
| Curia | Whole volume | Last phase | Image-only | No | 0.527 | 0.338 | 0.489 | 306 | curia_whole_last_phase_logreg |
| Curia | Whole volume | Last phase - phase 0 | Image-only | No | 0.533 | 0.345 | 0.518 | 306 | curia_whole_last_phase_minus_phase0_logreg |
| Curia | Whole volume | Phase 0 | Image-only | No | 0.480 | 0.306 | 0.497 | 306 | curia_whole_phase0_logreg |
| Curia | Whole volume | Phase 1 | Image-only | No | 0.501 | 0.306 | 0.487 | 306 | curia_whole_phase1_logreg |
| Curia | Whole volume | Phase 1 - phase 0 | Image-only | No | 0.473 | 0.287 | 0.498 | 306 | curia_whole_phase1_minus_phase0_logreg |
| Curia | Whole volume | Phase 2 | Image-only | No | 0.526 | 0.350 | 0.501 | 306 | curia_whole_phase2_logreg |
| Curia | Whole volume | Phase 2 - phase 0 | Image-only | No | 0.513 | 0.316 | 0.522 | 306 | curia_whole_phase2_minus_phase0_logreg |
| Curia | Whole volume | Raw phase fusion | Image-only fused embeddings | No | 0.543 | 0.341 | 0.513 | 306 | curia_whole_raw_phases_fusion_logreg |
| Curia | Whole volume | Subtraction fusion | Image-only fused embeddings | No | 0.483 | 0.306 | 0.463 | 306 | curia_whole_subtractions_fusion_logreg |
| Curia | Expert ROI | Phase 1 | Image-only | No | 0.602 | 0.430 | 0.543 | 306 | curia_expert_roi_phase1_logreg |
| Curia | Expert ROI | Phase 2 - phase 0 | Image-only | No | 0.630 | 0.434 | 0.578 | 306 | curia_expert_roi_phase2_minus_phase0_logreg |
| Curia | Expert ROI | Selected ROI fusion | Image-only fused embeddings | No | 0.608 | 0.446 | 0.546 | 306 | curia_expert_roi_selected_fusion_logreg |
| MedSigLIP | Whole volume | All DCE fusion | Image-only fused embeddings | No | 0.547 | 0.369 | 0.520 | 306 | medsiglip_whole_all_dce_fusion_logreg |
| MedSigLIP | Whole volume | Last phase | Image-only | No | 0.593 | 0.378 | 0.546 | 306 | medsiglip_whole_last_phase_logreg |
| MedSigLIP | Whole volume | Last phase - phase 0 | Image-only | No | 0.561 | 0.347 | 0.542 | 306 | medsiglip_whole_last_phase_minus_phase0_logreg |
| MedSigLIP | Whole volume | Phase 0 | Image-only | No | 0.541 | 0.331 | 0.541 | 306 | medsiglip_whole_phase0_logreg |
| MedSigLIP | Whole volume | Phase 1 | Image-only | No | 0.543 | 0.366 | 0.533 | 306 | medsiglip_whole_phase1_logreg |
| MedSigLIP | Whole volume | Phase 1 - phase 0 | Image-only | No | 0.587 | 0.363 | 0.562 | 306 | medsiglip_whole_phase1_minus_phase0_logreg |
| MedSigLIP | Whole volume | Phase 2 | Image-only | No | 0.590 | 0.389 | 0.542 | 306 | medsiglip_whole_phase2_logreg |
| MedSigLIP | Whole volume | Phase 2 - phase 0 | Image-only | No | 0.585 | 0.390 | 0.544 | 306 | medsiglip_whole_phase2_minus_phase0_logreg |
| MedSigLIP | Whole volume | Raw phase fusion | Image-only fused embeddings | No | 0.546 | 0.371 | 0.529 | 306 | medsiglip_whole_raw_phases_fusion_logreg |
| MedSigLIP | Whole volume | Subtraction fusion | Image-only fused embeddings | No | 0.565 | 0.347 | 0.555 | 306 | medsiglip_whole_subtractions_fusion_logreg |
| MedSigLIP | Expert ROI | Phase 1 | Image-only | No | 0.576 | 0.379 | 0.551 | 306 | medsiglip_expert_roi_phase1_logreg |
| MedSigLIP | Expert ROI | Phase 2 - phase 0 | Image-only | No | 0.621 | 0.403 | 0.565 | 306 | medsiglip_expert_roi_phase2_minus_phase0_logreg |
| MedSigLIP | Expert ROI | Selected ROI fusion | Image-only fused embeddings | No | 0.619 | 0.404 | 0.578 | 306 | medsiglip_expert_roi_selected_fusion_logreg |

## Image + clinical

| model | crop | input | feature_set | Clinical | AUROC | AP | Bal Acc | official_test | Run |
|---|---|---|---|---|---:|---:|---:|---|---|
| RadioDINO | Whole volume | All DCE fusion | Image + clinical | Yes | 0.614 | 0.405 | 0.602 | 306 | radiodino_whole_all_dce_fusion_image_clinical_logreg |
| RadioDINO | Whole volume | Phase 1 | Image + clinical | Yes | 0.640 | 0.407 | 0.592 | 306 | radiodino_whole_phase1_image_clinical_logreg |
| RadioDINO | Whole volume | Phase 2 - phase 0 | Image + clinical | Yes | 0.701 | 0.475 | 0.622 | 306 | radiodino_whole_phase2_minus_phase0_image_clinical_logreg |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | Image + clinical | Yes | 0.706 | 0.501 | 0.638 | 306 | radiodino_expert_roi_phase2_minus_phase0_image_clinical_logreg |
| BiomedCLIP | Whole volume | All DCE fusion | Image + clinical | Yes | 0.670 | 0.468 | 0.624 | 306 | biomedclip_whole_all_dce_fusion_image_clinical_logreg |
| BiomedCLIP | Whole volume | Phase 1 | Image + clinical | Yes | 0.739 | 0.553 | 0.605 | 306 | biomedclip_whole_phase1_image_clinical_logreg |
| BiomedCLIP | Whole volume | Phase 2 - phase 0 | Image + clinical | Yes | 0.686 | 0.452 | 0.613 | 306 | biomedclip_whole_phase2_minus_phase0_image_clinical_logreg |
| BiomedCLIP | Expert ROI | Phase 2 - phase 0 | Image + clinical | Yes | 0.706 | 0.513 | 0.652 | 306 | biomedclip_expert_roi_phase2_minus_phase0_image_clinical_logreg |
| Curia | Whole volume | All DCE fusion | Image + clinical | Yes | 0.621 | 0.400 | 0.617 | 306 | curia_whole_all_dce_fusion_image_clinical_logreg |
| Curia | Whole volume | Phase 1 | Image + clinical | Yes | 0.673 | 0.443 | 0.582 | 306 | curia_whole_phase1_image_clinical_logreg |
| Curia | Whole volume | Phase 2 - phase 0 | Image + clinical | Yes | 0.653 | 0.445 | 0.588 | 306 | curia_whole_phase2_minus_phase0_image_clinical_logreg |
| Curia | Expert ROI | Phase 2 - phase 0 | Image + clinical | Yes | 0.724 | 0.543 | 0.658 | 306 | curia_expert_roi_phase2_minus_phase0_image_clinical_logreg |
| MedSigLIP | Whole volume | All DCE fusion | Image + clinical | Yes | 0.639 | 0.426 | 0.621 | 306 | medsiglip_whole_all_dce_fusion_image_clinical_logreg |
| MedSigLIP | Whole volume | Phase 1 | Image + clinical | Yes | 0.706 | 0.483 | 0.579 | 306 | medsiglip_whole_phase1_image_clinical_logreg |
| MedSigLIP | Whole volume | Phase 2 - phase 0 | Image + clinical | Yes | 0.699 | 0.482 | 0.632 | 306 | medsiglip_whole_phase2_minus_phase0_image_clinical_logreg |
| MedSigLIP | Expert ROI | Phase 2 - phase 0 | Image + clinical | Yes | 0.703 | 0.481 | 0.653 | 306 | medsiglip_expert_roi_phase2_minus_phase0_image_clinical_logreg |
