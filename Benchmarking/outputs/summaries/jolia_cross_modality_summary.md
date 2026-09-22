# Jolia CT-to-MRI Transfer Stress Test

## Scope

Jolia is a 3D foundation model trained on adult chest and abdominal CT. This
experiment tests whether its frozen global representation transfers to breast
DCE-MRI pCR prediction. It is a cross-modality stress test and is intentionally
excluded from the modality-matched primary leaderboard.

The model revision was pinned to
`261abf87b0b1d77e74a75ecf0d6e0ca04043fb01`. MRI volumes were robustly scaled
to `[0, 1]`, fit into a `192 x 192 x 192` cube while preserving aspect ratio,
and repeated across Jolia's 11 fixed CT-window channels. Only the 576-D global
embedding was used. The official MAMA-MIA split contained 1,185 training and
306 test patients; every extraction covered 1,491 patients with zero failures.

## Results

| Features | Crop | Clinical | AUROC | AP | Balanced accuracy |
|---|---|---:|---:|---:|---:|
| Phase 1 | Whole volume | No | 0.491 | 0.305 | 0.475 |
| Phase 2 - phase 0 | Whole volume | No | 0.489 | 0.321 | 0.515 |
| Phase 2 - phase 0 | Expert ROI | No | 0.539 | 0.335 | 0.522 |
| Phase 1 + phase 2 subtraction fusion | Whole volume | No | 0.551 | 0.347 | 0.555 |
| Phase 1 | Whole volume | Yes | 0.635 | 0.424 | 0.593 |
| Phase 2 - phase 0 | Whole volume | Yes | 0.669 | 0.431 | 0.535 |
| Phase 2 - phase 0 | Expert ROI | Yes | **0.672** | **0.462** | **0.615** |
| Phase 1 + phase 2 subtraction fusion | Whole volume | Yes | 0.605 | 0.414 | 0.551 |
| Clinical-only reference | N/A | Yes | 0.735 | 0.502 | 0.642 |

AP should be interpreted against a test-set pCR prevalence of approximately
0.304.

## Interpretation

The two single whole-volume image embeddings are effectively at chance. Tumor
localization and phase fusion improve performance modestly, but the best
image-only result remains weak. Adding clinical variables produces much larger
gains, yet every Jolia image-plus-clinical configuration remains below the
clinical-only baseline on AUROC, AP, and balanced accuracy.

The result is consistent with the expected modality and anatomical domain
mismatch: a chest/abdominal CT representation does not provide a competitive
frozen representation for breast DCE-MRI pCR prediction under this transparent
input adapter. Jolia should be reported as a useful negative transfer control,
not as evidence against foundation models generally and not as a primary MRI
leaderboard entrant.

Detailed machine-readable results are in:

```text
Benchmarking/outputs/summaries/jolia_cross_modality_probe_summary.csv
```
