# Pillar-0 BreastMRI Whole-Volume Summary

Snapshot date: 2026-09-15

Pillar-0 whole-volume image-only benchmarking is complete for the four
3-channel 3D DCE input roles. Each extraction embedded 1491 patients with zero
failures and produced 1152-dimensional patient embeddings.

| Input | AUROC | AP | Balanced accuracy | Output |
|---|---:|---:|---:|---|
| Phase 0 + phase 1 + phase 2 | 0.542 | 0.366 | 0.531 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_phase2_logreg` |
| Phase 0 + phase 1 + last phase | 0.526 | 0.357 | 0.523 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_last_logreg` |
| Phase 0 + phase 2 + last phase | 0.546 | 0.358 | 0.527 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase2_last_logreg` |
| Subtraction triplet | 0.486 | 0.318 | 0.494 | `Benchmarking/outputs/probes/pillar0_breastmri_whole_subtractions_phase1_phase2_last_logreg` |

Interpretation: whole-volume Pillar-0 embeddings show weak pCR signal in this
first pass. The best whole-volume AUROC is 0.546, which is below the stronger
2D foundation-model image-only results and far below the clinical-only AUROC of
0.735. The next fair test is expert-ROI Pillar-0 extraction, because prior
models often improved when embeddings focused on the tumor region.
