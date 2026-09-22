# Foundation-Model Shortlist Bootstrap Summary

Snapshot date: 2026-09-22

Intervals use 5,000 paired bootstrap resamples (seed 20260921)
stratified jointly by source dataset and pCR class. Balanced accuracy
uses each probe's threshold selected on its training-only validation split.

## Important Scope

The shortlist was selected after inspecting official-test performance. These
95% intervals are descriptive uncertainty estimates, not confirmatory
post-selection significance tests. They do not correct for trying multiple
models, inputs, crops, or fusion variants.

## Shortlist Intervals

| Model | Group | Crop | Input | AUROC | AP | Bal Acc |
|---|---|---|---|---|---|---|
| Clinical baseline | Clinical-only | N/A | Clinical variables | 0.735 [0.681, 0.786] | 0.502 [0.438, 0.599] | 0.642 [0.591, 0.694] |
| BiomedCLIP | Image-only | Whole volume | Phase 1 | 0.618 [0.556, 0.679] | 0.404 [0.350, 0.488] | 0.576 [0.526, 0.625] |
| Curia | Image-only | Expert ROI | Phase 2 - phase 0 | 0.630 [0.564, 0.697] | 0.434 [0.367, 0.529] | 0.578 [0.523, 0.636] |
| Curia | Image-only | Expert ROI | Selected ROI fusion | 0.608 [0.541, 0.674] | 0.446 [0.373, 0.528] | 0.546 [0.502, 0.592] |
| MedSigLIP | Image-only | Expert ROI | Phase 2 - phase 0 | 0.621 [0.555, 0.683] | 0.403 [0.348, 0.486] | 0.565 [0.514, 0.616] |
| Pillar-0 BreastMRI | Image-only | Expert ROI | Phase 0 + phase 2 + last phase | 0.586 [0.513, 0.657] | 0.403 [0.340, 0.493] | 0.563 [0.503, 0.622] |
| RadImageNet | Image-only | Whole volume | Phase 2 - phase 0 | 0.619 [0.554, 0.683] | 0.396 [0.343, 0.479] | 0.593 [0.533, 0.650] |
| RadioDINO | Image-only | Expert ROI | Phase 2 - phase 0 | 0.573 [0.505, 0.638] | 0.387 [0.326, 0.463] | 0.554 [0.495, 0.614] |
| BiomedCLIP | Image + clinical | Whole volume | Phase 1 | 0.739 [0.685, 0.790] | 0.553 [0.477, 0.634] | 0.605 [0.552, 0.657] |
| Curia | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.724 [0.664, 0.782] | 0.543 [0.467, 0.646] | 0.658 [0.601, 0.715] |
| MedSigLIP | Image + clinical | Whole volume | Phase 1 | 0.706 [0.650, 0.758] | 0.483 [0.419, 0.567] | 0.579 [0.530, 0.629] |
| Pillar-0 BreastMRI | Image + clinical | Expert ROI | Phase 0 + phase 1 + last phase | 0.641 [0.572, 0.707] | 0.481 [0.403, 0.571] | 0.592 [0.534, 0.650] |
| RadImageNet | Image + clinical | Whole volume | Phase 2 - phase 0 | 0.725 [0.663, 0.785] | 0.510 [0.440, 0.612] | 0.681 [0.623, 0.736] |
| RadioDINO | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.706 [0.643, 0.766] | 0.501 [0.429, 0.596] | 0.638 [0.581, 0.694] |

## Image Plus Clinical Minus Clinical-Only

Positive values favor image plus clinical. Every difference uses the same
resampled patients for both models.

| Model | Crop | Input | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---|---|---|---|---|
| BiomedCLIP | Whole volume | Phase 1 | 0.004 [-0.037, 0.045] | 0.050 [-0.030, 0.111] | -0.037 [-0.092, 0.017] |
| Curia | Expert ROI | Phase 2 - phase 0 | -0.011 [-0.070, 0.051] | 0.041 [-0.059, 0.137] | 0.016 [-0.049, 0.082] |
| MedSigLIP | Whole volume | Phase 1 | -0.030 [-0.078, 0.020] | -0.019 [-0.109, 0.059] | -0.062 [-0.117, -0.008] |
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | -0.094 [-0.168, -0.023] | -0.022 [-0.133, 0.076] | -0.049 [-0.118, 0.017] |
| RadImageNet | Whole volume | Phase 2 - phase 0 | -0.010 [-0.070, 0.050] | 0.008 [-0.077, 0.094] | 0.039 [-0.022, 0.100] |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | -0.029 [-0.079, 0.022] | -0.002 [-0.093, 0.079] | -0.003 [-0.057, 0.050] |

## Interpretation Guardrails

- An interval crossing zero means the paired bootstrap does not show a stable
  direction of difference at this sample size.
- An interval excluding zero is still exploratory because configurations were
  compared and shortlisted using this same test set.
- Final model selection or tuning should return to training-only nested
  validation or use a genuinely untouched external cohort.
