# BiomedCLIP First-Pass Benchmark Summary

Snapshot date: 2026-08-31

Full metric table:

```text
Benchmarking/outputs/summaries/biomedclip_first_pass_probe_summary.csv
```

## Command Completion Check

The BiomedCLIP first-pass benchmark completed successfully.

All expected embedding tables were found:

- whole-volume phase0
- whole-volume phase1
- whole-volume phase2
- whole-volume last phase
- whole-volume phase1-minus-phase0
- whole-volume phase2-minus-phase0
- whole-volume last-phase-minus-phase0
- whole-volume raw phase fusion
- whole-volume subtraction fusion
- whole-volume all-DCE fusion
- expert-ROI phase1
- expert-ROI phase2-minus-phase0
- expert-ROI selected fusion

All embedding jobs contain 1491 embedded patients and 0 failures. Single-input
BiomedCLIP embeddings have 512 features per patient. Raw phase fusion has 2048
features, subtraction fusion has 1536 features, all-DCE fusion has 3584
features, and selected ROI fusion has 1024 features.

All expected probe outputs were found.

## Key Results

| Experiment | Crop | Feature Set | Clinical Data | Test AUROC | Test AP | Test Bal Acc |
|---|---|---|---|---:|---:|---:|
| Clinical-only baseline | N/A | Clinical variables | Yes | 0.735 | 0.502 | 0.642 |
| BiomedCLIP phase0 | Whole volume | Image-only | No | 0.572 | 0.383 | 0.535 |
| BiomedCLIP phase1 | Whole volume | Image-only | No | 0.618 | 0.404 | 0.576 |
| BiomedCLIP phase2 | Whole volume | Image-only | No | 0.584 | 0.397 | 0.536 |
| BiomedCLIP last phase | Whole volume | Image-only | No | 0.591 | 0.381 | 0.567 |
| BiomedCLIP phase1 - phase0 | Whole volume | Image-only | No | 0.604 | 0.382 | 0.529 |
| BiomedCLIP phase2 - phase0 | Whole volume | Image-only | No | 0.542 | 0.341 | 0.519 |
| BiomedCLIP last phase - phase0 | Whole volume | Image-only | No | 0.546 | 0.352 | 0.504 |
| BiomedCLIP raw phase fusion | Whole volume | Image-only fusion | No | 0.583 | 0.371 | 0.560 |
| BiomedCLIP subtraction fusion | Whole volume | Image-only fusion | No | 0.556 | 0.364 | 0.530 |
| BiomedCLIP all DCE fusion | Whole volume | Image-only fusion | No | 0.602 | 0.409 | 0.544 |
| BiomedCLIP phase1 | Expert ROI | Image-only | No | 0.580 | 0.380 | 0.561 |
| BiomedCLIP phase2 - phase0 | Expert ROI | Image-only | No | 0.549 | 0.363 | 0.549 |
| BiomedCLIP selected ROI fusion | Expert ROI | Image-only fusion | No | 0.565 | 0.361 | 0.531 |
| BiomedCLIP phase1 + clinical | Whole volume | Image + clinical | Yes | 0.739 | 0.553 | 0.605 |
| BiomedCLIP phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | 0.686 | 0.452 | 0.613 |
| BiomedCLIP all DCE fusion + clinical | Whole volume | Image + clinical | Yes | 0.670 | 0.468 | 0.624 |
| BiomedCLIP phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | 0.706 | 0.513 | 0.652 |

## Interpretation

BiomedCLIP is clearly stronger than RadioDINO for image-only pCR prediction in
this first-pass benchmark. Its best single image-only result is whole-volume
phase1, with AUROC 0.618, AP 0.404, and balanced accuracy 0.576. By comparison,
the best RadioDINO image-only result was expert-ROI phase2-minus-phase0, with
AUROC 0.573, AP 0.387, and balanced accuracy 0.554.

Simple multiphase feature fusion does not clearly improve BiomedCLIP. All-DCE
fusion has the best image-only AP at 0.409, but its AUROC is lower than phase1
alone. Raw phase fusion and subtraction fusion are also below the best single
phase.

ROI cropping does not help BiomedCLIP as much as it helped RadioDINO. Whole
phase1 remains better than ROI phase1. ROI phase2-minus-phase0 is only slightly
better than whole phase2-minus-phase0 by AP and balanced accuracy, but not by
enough to make ROI the best image-only input. Selected ROI fusion is also
weaker than the best single whole-volume phase.

The strongest multimodal BiomedCLIP result by ranking metrics is whole phase1
plus clinical: AUROC 0.739 and AP 0.553. This is the first benchmark result that
beats the clinical-only baseline on both AUROC and AP. The strongest selected
threshold balanced accuracy is BiomedCLIP expert-ROI phase2-minus-phase0 plus
clinical at 0.652, compared with 0.642 for clinical-only.

## Conclusion

BiomedCLIP should be kept as a stronger baseline than RadioDINO. The best
first-pass configurations are:

```text
image-only:          whole phase1
image-only by AP:    whole all-DCE fusion
image + clinical:    whole phase1 + clinical
balanced threshold:  expert ROI phase2_minus_phase0 + clinical
```

The most important scientific result so far is that BiomedCLIP image embeddings
appear to add useful information beyond clinical variables, especially for
ranking patients by pCR probability.
