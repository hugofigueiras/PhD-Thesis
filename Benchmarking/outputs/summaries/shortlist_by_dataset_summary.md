# Foundation-Model Shortlist By Source Dataset

Snapshot date: 2026-09-22

Results use 5,000 paired bootstrap resamples per source dataset
(seed 20260922), stratified by pCR class. Intervals are 95%
percentile intervals.

## Scope

This is a subgroup robustness analysis of the official test split, not an
external generalization experiment. Each source dataset also contributes
patients to the official training split. The shortlist was selected after
inspecting aggregate test performance, so all intervals are exploratory.

## Test Composition

| Dataset | N | pCR | Non-pCR | pCR prevalence |
|---|---|---|---|---|
| DUKE | 91 | 30 | 61 | 0.330 |
| ISPY1 | 67 | 14 | 53 | 0.209 |
| ISPY2 | 131 | 46 | 85 | 0.351 |
| NACT | 17 | 3 | 14 | 0.176 |

NACT has only three positive test cases; its intervals are especially
unstable and should not be used for model ranking.

## DUKE

Test patients: 91; pCR cases: 30.

| Model | Group | Crop | Input | AUROC | AP | Bal Acc |
|---|---|---|---|---|---|---|
| Clinical baseline | Clinical-only | N/A | Clinical variables | 0.787 [0.692, 0.872] | 0.668 [0.545, 0.796] | 0.592 [0.525, 0.667] |
| BiomedCLIP | Image-only | Whole volume | Phase 1 | 0.495 [0.362, 0.626] | 0.376 [0.289, 0.513] | 0.538 [0.429, 0.645] |
| Curia | Image-only | Expert ROI | Phase 2 - phase 0 | 0.623 [0.496, 0.746] | 0.487 [0.371, 0.643] | 0.552 [0.460, 0.651] |
| Curia | Image-only | Expert ROI | Selected ROI fusion | 0.634 [0.515, 0.752] | 0.462 [0.359, 0.622] | 0.526 [0.451, 0.602] |
| MedSigLIP | Image-only | Expert ROI | Phase 2 - phase 0 | 0.690 [0.574, 0.802] | 0.480 [0.389, 0.654] | 0.526 [0.459, 0.593] |
| Pillar-0 BreastMRI | Image-only | Expert ROI | Phase 0 + phase 2 + last phase | 0.604 [0.479, 0.727] | 0.426 [0.340, 0.602] | 0.561 [0.454, 0.669] |
| RadImageNet | Image-only | Whole volume | Phase 2 - phase 0 | 0.620 [0.501, 0.737] | 0.429 [0.342, 0.574] | 0.585 [0.486, 0.685] |
| RadioDINO | Image-only | Expert ROI | Phase 2 - phase 0 | 0.587 [0.473, 0.702] | 0.367 [0.310, 0.505] | 0.528 [0.421, 0.636] |
| BiomedCLIP | Image + clinical | Whole volume | Phase 1 | 0.703 [0.590, 0.807] | 0.542 [0.421, 0.695] | 0.584 [0.509, 0.667] |
| Curia | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.735 [0.622, 0.837] | 0.626 [0.493, 0.772] | 0.693 [0.594, 0.793] |
| MedSigLIP | Image + clinical | Whole volume | Phase 1 | 0.738 [0.626, 0.838] | 0.585 [0.458, 0.737] | 0.542 [0.492, 0.600] |
| Pillar-0 BreastMRI | Image + clinical | Expert ROI | Phase 0 + phase 1 + last phase | 0.679 [0.555, 0.796] | 0.508 [0.402, 0.697] | 0.661 [0.554, 0.761] |
| RadImageNet | Image + clinical | Whole volume | Phase 2 - phase 0 | 0.752 [0.636, 0.858] | 0.660 [0.530, 0.801] | 0.718 [0.619, 0.818] |
| RadioDINO | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.754 [0.653, 0.851] | 0.559 [0.449, 0.752] | 0.643 [0.551, 0.743] |

### Image Plus Clinical Minus Clinical-Only

| Model | Crop | Input | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---|---|---|---|---|
| BiomedCLIP | Whole volume | Phase 1 | -0.084 [-0.171, 0.001] | -0.126 [-0.240, -0.002] | -0.008 [-0.091, 0.067] |
| Curia | Expert ROI | Phase 2 - phase 0 | -0.052 [-0.157, 0.050] | -0.042 [-0.172, 0.095] | 0.101 [0.010, 0.201] |
| MedSigLIP | Whole volume | Phase 1 | -0.049 [-0.147, 0.047] | -0.083 [-0.199, 0.045] | -0.050 [-0.125, 0.017] |
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | -0.108 [-0.242, 0.028] | -0.160 [-0.317, 0.055] | 0.069 [-0.055, 0.193] |
| RadImageNet | Whole volume | Phase 2 - phase 0 | -0.035 [-0.139, 0.070] | -0.008 [-0.147, 0.133] | 0.127 [0.018, 0.235] |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | -0.033 [-0.111, 0.048] | -0.109 [-0.225, 0.063] | 0.051 [-0.032, 0.142] |

## ISPY1

Test patients: 67; pCR cases: 14.

| Model | Group | Crop | Input | AUROC | AP | Bal Acc |
|---|---|---|---|---|---|---|
| Clinical baseline | Clinical-only | N/A | Clinical variables | 0.760 [0.621, 0.879] | 0.451 [0.311, 0.714] | 0.703 [0.565, 0.838] |
| BiomedCLIP | Image-only | Whole volume | Phase 1 | 0.617 [0.468, 0.761] | 0.268 [0.210, 0.445] | 0.584 [0.439, 0.717] |
| Curia | Image-only | Expert ROI | Phase 2 - phase 0 | 0.577 [0.419, 0.733] | 0.267 [0.200, 0.481] | 0.520 [0.394, 0.654] |
| Curia | Image-only | Expert ROI | Selected ROI fusion | 0.477 [0.313, 0.651] | 0.241 [0.173, 0.465] | 0.481 [0.453, 0.500] |
| MedSigLIP | Image-only | Expert ROI | Phase 2 - phase 0 | 0.530 [0.354, 0.706] | 0.257 [0.184, 0.467] | 0.528 [0.392, 0.670] |
| Pillar-0 BreastMRI | Image-only | Expert ROI | Phase 0 + phase 2 + last phase | 0.534 [0.363, 0.709] | 0.289 [0.190, 0.528] | 0.526 [0.383, 0.670] |
| RadImageNet | Image-only | Whole volume | Phase 2 - phase 0 | 0.655 [0.497, 0.799] | 0.306 [0.229, 0.530] | 0.618 [0.473, 0.761] |
| RadioDINO | Image-only | Expert ROI | Phase 2 - phase 0 | 0.470 [0.302, 0.643] | 0.249 [0.172, 0.474] | 0.460 [0.315, 0.602] |
| BiomedCLIP | Image + clinical | Whole volume | Phase 1 | 0.783 [0.666, 0.888] | 0.405 [0.302, 0.645] | 0.577 [0.460, 0.710] |
| Curia | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.759 [0.616, 0.880] | 0.518 [0.337, 0.735] | 0.613 [0.479, 0.748] |
| MedSigLIP | Image + clinical | Whole volume | Phase 1 | 0.691 [0.532, 0.842] | 0.350 [0.252, 0.600] | 0.603 [0.475, 0.744] |
| Pillar-0 BreastMRI | Image + clinical | Expert ROI | Phase 0 + phase 1 + last phase | 0.605 [0.430, 0.774] | 0.346 [0.222, 0.613] | 0.526 [0.381, 0.670] |
| RadImageNet | Image + clinical | Whole volume | Phase 2 - phase 0 | 0.751 [0.621, 0.865] | 0.417 [0.282, 0.633] | 0.644 [0.501, 0.787] |
| RadioDINO | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.698 [0.543, 0.830] | 0.355 [0.250, 0.592] | 0.590 [0.445, 0.732] |

### Image Plus Clinical Minus Clinical-Only

| Model | Crop | Input | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---|---|---|---|---|
| BiomedCLIP | Whole volume | Phase 1 | 0.023 [-0.070, 0.127] | -0.046 [-0.232, 0.129] | -0.126 [-0.252, -0.026] |
| Curia | Expert ROI | Phase 2 - phase 0 | -0.001 [-0.135, 0.137] | 0.067 [-0.173, 0.251] | -0.090 [-0.261, 0.090] |
| MedSigLIP | Whole volume | Phase 1 | -0.069 [-0.208, 0.085] | -0.101 [-0.336, 0.139] | -0.100 [-0.271, 0.079] |
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | -0.155 [-0.318, 0.016] | -0.105 [-0.351, 0.147] | -0.177 [-0.320, -0.040] |
| RadImageNet | Whole volume | Phase 2 - phase 0 | -0.009 [-0.136, 0.123] | -0.034 [-0.251, 0.127] | -0.059 [-0.187, 0.077] |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | -0.062 [-0.175, 0.058] | -0.096 [-0.306, 0.100] | -0.113 [-0.230, -0.004] |

## ISPY2

Test patients: 131; pCR cases: 46.

| Model | Group | Crop | Input | AUROC | AP | Bal Acc |
|---|---|---|---|---|---|---|
| Clinical baseline | Clinical-only | N/A | Clinical variables | 0.687 [0.591, 0.775] | 0.498 [0.413, 0.641] | 0.636 [0.550, 0.724] |
| BiomedCLIP | Image-only | Whole volume | Phase 1 | 0.713 [0.618, 0.804] | 0.560 [0.461, 0.705] | 0.591 [0.530, 0.649] |
| Curia | Image-only | Expert ROI | Phase 2 - phase 0 | 0.651 [0.549, 0.746] | 0.506 [0.412, 0.648] | 0.613 [0.524, 0.702] |
| Curia | Image-only | Expert ROI | Selected ROI fusion | 0.638 [0.536, 0.737] | 0.551 [0.444, 0.670] | 0.581 [0.502, 0.659] |
| MedSigLIP | Image-only | Expert ROI | Phase 2 - phase 0 | 0.685 [0.585, 0.780] | 0.571 [0.465, 0.695] | 0.628 [0.546, 0.710] |
| Pillar-0 BreastMRI | Image-only | Expert ROI | Phase 0 + phase 2 + last phase | 0.603 [0.500, 0.707] | 0.480 [0.387, 0.611] | 0.583 [0.495, 0.670] |
| RadImageNet | Image-only | Whole volume | Phase 2 - phase 0 | 0.630 [0.527, 0.731] | 0.488 [0.395, 0.621] | 0.599 [0.514, 0.685] |
| RadioDINO | Image-only | Expert ROI | Phase 2 - phase 0 | 0.633 [0.533, 0.734] | 0.519 [0.421, 0.637] | 0.625 [0.541, 0.710] |
| BiomedCLIP | Image + clinical | Whole volume | Phase 1 | 0.786 [0.704, 0.861] | 0.686 [0.582, 0.790] | 0.634 [0.551, 0.716] |
| Curia | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.707 [0.610, 0.799] | 0.570 [0.467, 0.716] | 0.632 [0.545, 0.717] |
| MedSigLIP | Image + clinical | Whole volume | Phase 1 | 0.741 [0.655, 0.823] | 0.570 [0.469, 0.700] | 0.601 [0.518, 0.684] |
| Pillar-0 BreastMRI | Image + clinical | Expert ROI | Phase 0 + phase 1 + last phase | 0.650 [0.546, 0.750] | 0.576 [0.464, 0.699] | 0.574 [0.486, 0.660] |
| RadImageNet | Image + clinical | Whole volume | Phase 2 - phase 0 | 0.706 [0.610, 0.798] | 0.581 [0.477, 0.723] | 0.660 [0.576, 0.740] |
| RadioDINO | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.714 [0.613, 0.809] | 0.600 [0.488, 0.742] | 0.675 [0.586, 0.759] |

### Image Plus Clinical Minus Clinical-Only

| Model | Crop | Input | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---|---|---|---|---|
| BiomedCLIP | Whole volume | Phase 1 | 0.099 [0.026, 0.177] | 0.188 [0.073, 0.267] | -0.002 [-0.095, 0.091] |
| Curia | Expert ROI | Phase 2 - phase 0 | 0.020 [-0.091, 0.131] | 0.072 [-0.075, 0.216] | -0.004 [-0.112, 0.107] |
| MedSigLIP | Whole volume | Phase 1 | 0.054 [-0.027, 0.142] | 0.072 [-0.056, 0.181] | -0.036 [-0.115, 0.046] |
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | -0.037 [-0.150, 0.079] | 0.078 [-0.084, 0.203] | -0.062 [-0.166, 0.044] |
| RadImageNet | Whole volume | Phase 2 - phase 0 | 0.019 [-0.085, 0.124] | 0.083 [-0.042, 0.207] | 0.023 [-0.071, 0.122] |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | 0.027 [-0.060, 0.121] | 0.102 [-0.022, 0.218] | 0.038 [-0.050, 0.126] |

## NACT

Test patients: 17; pCR cases: 3.

| Model | Group | Crop | Input | AUROC | AP | Bal Acc |
|---|---|---|---|---|---|---|
| Clinical baseline | Clinical-only | N/A | Clinical variables | 0.643 [0.262, 1.000] | 0.505 [0.190, 1.000] | 0.667 [0.500, 1.000] |
| BiomedCLIP | Image-only | Whole volume | Phase 1 | 0.524 [0.214, 0.857] | 0.233 [0.171, 0.535] | 0.476 [0.167, 0.714] |
| Curia | Image-only | Expert ROI | Phase 2 - phase 0 | 0.548 [0.214, 0.857] | 0.249 [0.175, 0.600] | 0.524 [0.286, 0.821] |
| Curia | Image-only | Expert ROI | Selected ROI fusion | 0.429 [0.143, 0.714] | 0.198 [0.156, 0.429] | 0.393 [0.286, 0.500] |
| MedSigLIP | Image-only | Expert ROI | Phase 2 - phase 0 | 0.095 [0.000, 0.286] | 0.132 [0.123, 0.205] | 0.393 [0.286, 0.500] |
| Pillar-0 BreastMRI | Image-only | Expert ROI | Phase 0 + phase 2 + last phase | 0.452 [0.000, 0.857] | 0.225 [0.155, 0.600] | 0.452 [0.214, 0.762] |
| RadImageNet | Image-only | Whole volume | Phase 2 - phase 0 | 0.619 [0.214, 0.952] | 0.349 [0.189, 0.867] | 0.512 [0.214, 0.786] |
| RadioDINO | Image-only | Expert ROI | Phase 2 - phase 0 | 0.048 [0.000, 0.144] | 0.127 [0.123, 0.188] | 0.286 [0.143, 0.393] |
| BiomedCLIP | Image + clinical | Whole volume | Phase 1 | 0.524 [0.214, 0.786] | 0.224 [0.171, 0.528] | 0.524 [0.286, 0.821] |
| Curia | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.571 [0.262, 0.857] | 0.269 [0.178, 0.750] | 0.560 [0.321, 0.857] |
| MedSigLIP | Image + clinical | Whole volume | Phase 1 | 0.357 [0.071, 0.690] | 0.179 [0.148, 0.424] | 0.464 [0.393, 0.500] |
| Pillar-0 BreastMRI | Image + clinical | Expert ROI | Phase 0 + phase 1 + last phase | 0.452 [0.167, 0.738] | 0.199 [0.159, 0.429] | 0.357 [0.249, 0.464] |
| RadImageNet | Image + clinical | Whole volume | Phase 2 - phase 0 | 0.786 [0.548, 0.976] | 0.387 [0.261, 0.917] | 0.786 [0.643, 0.929] |
| RadioDINO | Image + clinical | Expert ROI | Phase 2 - phase 0 | 0.381 [0.048, 0.786] | 0.201 [0.145, 0.544] | 0.393 [0.286, 0.500] |

### Image Plus Clinical Minus Clinical-Only

| Model | Crop | Input | Delta AUROC | Delta AP | Delta Bal Acc |
|---|---|---|---|---|---|
| BiomedCLIP | Whole volume | Phase 1 | -0.119 [-0.381, 0.143] | -0.281 [-0.535, 0.056] | -0.143 [-0.250, -0.036] |
| Curia | Expert ROI | Phase 2 - phase 0 | -0.071 [-0.357, 0.262] | -0.237 [-0.466, 0.089] | -0.107 [-0.214, 0.000] |
| MedSigLIP | Whole volume | Phase 1 | -0.286 [-0.476, -0.119] | -0.327 [-0.573, -0.026] | -0.202 [-0.500, 0.000] |
| Pillar-0 BreastMRI | Expert ROI | Phase 0 + phase 1 + last phase | -0.190 [-0.643, 0.262] | -0.306 [-0.750, 0.117] | -0.310 [-0.607, -0.071] |
| RadImageNet | Whole volume | Phase 2 - phase 0 | 0.143 [-0.167, 0.500] | -0.118 [-0.419, 0.402] | 0.119 [-0.190, 0.393] |
| RadioDINO | Expert ROI | Phase 2 - phase 0 | -0.262 [-0.476, -0.095] | -0.304 [-0.533, -0.025] | -0.274 [-0.571, -0.036] |

## Interpretation Guardrails

- Compare directions and interval widths across cohorts; do not rank
  models from a single small subgroup.
- The same validation-selected threshold is applied to every source
  subgroup, so balanced accuracy also checks threshold transportability.
- A proper cross-cohort generalization claim requires retraining the probe
  while holding an entire source dataset out from all model selection.
