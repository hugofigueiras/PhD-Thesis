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

## Current Status

Current snapshot: 73 completed probe runs, consisting of 1 clinical-only run,
56 image-only embedding runs, and 16 image-plus-clinical runs.

The matrix varies:

| Dimension | Values currently represented |
|---|---|
| Foundation model | Pillar-0 BreastMRI, RadioDINO, BiomedCLIP, Curia, MedSigLIP |
| Image crop | Whole volume, expert ROI |
| DCE input | Phase 0, phase 1, phase 2, last phase, post-contrast subtraction, raw phase fusion, subtraction fusion, all-DCE fusion, Pillar-0 3D phase triplets |
| Feature set | Clinical-only, image-only, image-plus-clinical |
| Classifier | L2-regularized logistic regression |

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
| 15 | Pillar-0 BreastMRI ROI and image + clinical | Whole volume and expert ROI | 3D image-only, image + clinical | Mixed | Pending | TBD | TBD | TBD | `Benchmarking/pillar0_benchmark_commands.md` |

## Interpretation So Far

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
fusion reaches AUROC 0.565, and all-DCE fusion reaches AUROC 0.547. The next
MedSigLIP step is expert ROI extraction, because ROI was where Curia and
RadioDINO improved most.

MedSigLIP expert ROI image-only benchmarking is also complete. ROI improves the
best MedSigLIP image-only AUROC from 0.593 to 0.621 using expert-ROI
phase2-minus-phase0. Selected ROI fusion has very similar AUROC at 0.619 and the
best MedSigLIP image-only AP/balanced accuracy at 0.404/0.578. Continue with the
standardized image-plus-clinical probes.

MedSigLIP first-pass benchmarking is complete. The best MedSigLIP multimodal
AUROC/AP comes from whole phase1 plus clinical at 0.706/0.483. The best
MedSigLIP multimodal balanced accuracy comes from expert-ROI
phase2-minus-phase0 plus clinical at 0.653. MedSigLIP improves substantially
when clinical variables are added, but it does not beat the current best
BiomedCLIP AUROC/AP or Curia balanced accuracy multimodal results.

Pillar-0 BreastMRI is the next benchmark target because it is a 3D breast MRI
foundation model rather than another 2D slice encoder. Its first-pass grid uses
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
prediction. The next fair check is expert ROI, because several previous models
improved after focusing the representation on the tumor region.

The complete result table is written to
`Benchmarking/outputs/summaries/cross_model_all_results.md`. This table should
be treated as the appendix/source table: it includes every completed run, while
the cross-model first-pass summary keeps the headline standardized comparisons.

## Commands To Run Next

The cross-model first-pass comparison has been written. Regenerate it after any
future benchmark run with:

```bash
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
```

To regenerate the global probe summary before rebuilding the comparison:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
```

MedSigLIP first-pass benchmarking is complete. Pillar-0 BreastMRI whole-volume
image-only benchmarking is complete. Continue with Pillar-0 expert-ROI
extraction if you want to complete the standardized Pillar-0 first pass.

```text
Benchmarking/pillar0_benchmark_commands.md
```

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

## Optional RadioDINO Expansion

If expanding RadioDINO, train ROI image plus clinical for another ROI embedding
table:

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/radiodino/expert_roi_<best>/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/radiodino_expert_roi_<best>_image_clinical_logreg
```
