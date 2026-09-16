# Curia First-Pass Benchmark Summary

Snapshot date: 2026-09-07

Curia was benchmarked with the same first-pass protocol used for RadioDINO and
BiomedCLIP: frozen image encoder, patient-level embeddings, L2 logistic probe,
official MAMA-MIA train/test split, and clinical fusion only after embedding
extraction.

Full metric tables:

```text
Benchmarking/outputs/summaries/curia_first_pass_probe_summary.csv
Benchmarking/outputs/summaries/probe_run_summary.csv
```

## Command Completion Check

The Curia first-pass benchmark completed successfully.

All expected Curia outputs were found:

- whole-volume phase0, phase1, phase2, and last phase
- whole-volume phase1-minus-phase0, phase2-minus-phase0, and last-phase-minus-phase0
- whole-volume raw phase fusion, subtraction fusion, and all-DCE fusion
- expert-ROI phase1 and phase2-minus-phase0
- expert-ROI selected fusion
- whole-volume image-plus-clinical probes
- expert-ROI phase2-minus-phase0 image-plus-clinical probe

Single-input Curia embeddings have 768 features per patient. Raw phase fusion
has 3072 features, subtraction fusion has 2304 features, all-DCE fusion has 5376
features, and selected ROI fusion has 1536 features.

## Key Results

| Experiment | Crop | Feature Set | Clinical Data | Test AUROC | Test AP | Test Bal Acc |
|---|---|---|---|---:|---:|---:|
| Clinical-only baseline | N/A | Clinical variables | Yes | 0.735 | 0.502 | 0.642 |
| Curia phase0 | Whole volume | Image-only | No | 0.480 | 0.306 | 0.497 |
| Curia phase1 | Whole volume | Image-only | No | 0.501 | 0.306 | 0.487 |
| Curia phase2 | Whole volume | Image-only | No | 0.526 | 0.350 | 0.501 |
| Curia last phase | Whole volume | Image-only | No | 0.527 | 0.338 | 0.489 |
| Curia phase1 - phase0 | Whole volume | Image-only | No | 0.473 | 0.287 | 0.498 |
| Curia phase2 - phase0 | Whole volume | Image-only | No | 0.513 | 0.316 | 0.522 |
| Curia last phase - phase0 | Whole volume | Image-only | No | 0.533 | 0.345 | 0.518 |
| Curia raw phase fusion | Whole volume | Image-only fusion | No | 0.543 | 0.341 | 0.513 |
| Curia subtraction fusion | Whole volume | Image-only fusion | No | 0.483 | 0.306 | 0.463 |
| Curia all-DCE fusion | Whole volume | Image-only fusion | No | 0.531 | 0.343 | 0.505 |
| Curia phase1 | Expert ROI | Image-only | No | 0.602 | 0.430 | 0.543 |
| Curia phase2 - phase0 | Expert ROI | Image-only | No | 0.630 | 0.434 | 0.578 |
| Curia selected ROI fusion | Expert ROI | Image-only fusion | No | 0.608 | 0.446 | 0.546 |
| Curia phase1 + clinical | Whole volume | Image + clinical | Yes | 0.673 | 0.443 | 0.582 |
| Curia phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | 0.653 | 0.445 | 0.588 |
| Curia all-DCE fusion + clinical | Whole volume | Image + clinical | Yes | 0.621 | 0.400 | 0.617 |
| Curia phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | 0.724 | 0.543 | 0.658 |

## Interpretation

Curia whole-volume embeddings are weak for pCR prediction. The best
whole-volume image-only AUROC is raw phase fusion at 0.543, and the best
whole-volume AP is phase2 at 0.350. Whole-volume all-DCE fusion does not improve
the signal, and subtraction fusion is worse than chance by AUROC.

Expert ROI cropping is the key Curia result. Curia expert-ROI
phase2-minus-phase0 reaches AUROC 0.630, AP 0.434, and balanced accuracy 0.578.
Selected ROI fusion gives the best Curia image-only AP at 0.446, but its AUROC
and balanced accuracy are lower than the single expert-ROI subtraction input.

Image-plus-clinical fusion only helps strongly when the image features come
from the expert ROI subtraction input. Whole-volume Curia plus clinical remains
below the clinical-only baseline. Curia expert-ROI phase2-minus-phase0 plus
clinical reaches AUROC 0.724, AP 0.543, and balanced accuracy 0.658.

## Cross-Model Position

The best Curia image-only result is stronger than the best RadioDINO image-only
result and slightly stronger than the best BiomedCLIP image-only result by AUROC
and AP.

The best Curia multimodal result is slightly below BiomedCLIP phase1 plus
clinical by AUROC and AP, but it has the best selected-threshold balanced
accuracy seen so far:

```text
Best AUROC/AP:      BiomedCLIP whole phase1 + clinical
Best Bal Acc:       Curia expert ROI phase2_minus_phase0 + clinical
Best Curia image:   expert ROI phase2_minus_phase0
Curia keep/drop:    keep Curia, but only as an ROI-focused model
```

## Conclusion

Curia should not be judged by whole-volume performance alone. As a whole-volume
encoder it is weak, but with expert ROI cropping it becomes one of the strongest
image-only models tested so far. The most defensible Curia configuration for
later comparison is expert-ROI phase2-minus-phase0, with and without clinical
fusion.
