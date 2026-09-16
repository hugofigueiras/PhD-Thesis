# RadioDINO First-Pass Benchmark Summary

Snapshot date: 2026-08-28

Full metric table:

```text
Benchmarking/outputs/summaries/radiodino_first_pass_probe_summary.csv
```

## Benchmark Setup

- Dataset: MAMA-MIA official train/test split.
- Target: binary pCR.
- Foundation model: frozen RadioDINO image encoder.
- Imaging protocol: 2D axial slice embeddings aggregated to one patient-level
  vector.
- Probe: L2-regularized logistic regression.
- Hyperparameter selection: inner validation split from official train only.
- Test set: official MAMA-MIA test split only.

## Key Results

| Experiment | Crop | Feature Set | Clinical Data | Test AUROC | Test AP | Test Bal Acc |
|---|---|---|---|---:|---:|---:|
| Clinical-only baseline | N/A | Clinical variables | Yes | 0.735 | 0.502 | 0.642 |
| RadioDINO phase1 | Whole volume | Image-only | No | 0.547 | 0.364 | 0.528 |
| RadioDINO phase2 - phase0 | Whole volume | Image-only | No | 0.549 | 0.327 | 0.553 |
| RadioDINO all DCE fusion | Whole volume | Image-only fusion | No | 0.548 | 0.351 | 0.542 |
| RadioDINO phase1 | Expert ROI | Image-only | No | 0.546 | 0.376 | 0.522 |
| RadioDINO phase2 - phase0 | Expert ROI | Image-only | No | 0.573 | 0.387 | 0.554 |
| RadioDINO selected ROI fusion | Expert ROI | Image-only fusion | No | 0.556 | 0.364 | 0.536 |
| RadioDINO phase2 - phase0 + clinical | Whole volume | Image + clinical | Yes | 0.701 | 0.475 | 0.622 |
| RadioDINO phase2 - phase0 ROI + clinical | Expert ROI | Image + clinical | Yes | 0.706 | 0.501 | 0.638 |

## Interpretation

RadioDINO image-only embeddings contain weak pCR signal on this benchmark.
Expert ROI cropping modestly improves the best subtraction experiment:
phase2-minus-phase0 rises from 0.549 AUROC and 0.327 AP with whole volumes to
0.573 AUROC and 0.387 AP with expert ROI crops.

Simple ROI feature fusion does not improve over the single best ROI subtraction
embedding. The selected ROI fusion result is lower than ROI phase2-minus-phase0
alone.

Image plus clinical fusion gives the best RadioDINO multimodal result, but it
does not clearly beat clinical-only. ROI phase2-minus-phase0 plus clinical has
nearly identical AP to clinical-only, slightly lower balanced accuracy, and
lower AUROC. It detects more pCR cases than clinical-only, but with more false
positives.

## Conclusion

RadioDINO should be recorded as completed for the first-pass benchmark. The
strongest RadioDINO configuration is:

```text
expert ROI phase2_minus_phase0 + clinical
```

However, the foundation-model image representation does not yet add a clear
performance gain over the clinical-only baseline. The next benchmark should
move to another foundation model instead of expanding RadioDINO further.

## Next Foundation Model

Recommended next model: BiomedCLIP.

Reason: BiomedCLIP is a public OpenCLIP-compatible biomedical vision-language
model, which makes it a practical next benchmark. It is less breast-MRI-specific
than Pillar-0 or Curia, but it is useful as a broad biomedical control and can
reuse the same 2D slice aggregation protocol used for RadioDINO.
