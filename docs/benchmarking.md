# Foundation-Model Benchmarking

The current benchmark evaluates whether frozen medical foundation-model
representations contain useful signal for pCR prediction, and whether they add
predictive value beyond tabular clinical variables.

## Current Benchmark Dataset

The current benchmark uses the original MAMA-MIA official train/test split and
the pretreatment DCE-MRI exam. Longitudinal reconstructed timepoints are the next
dataset extension, but the first-pass benchmark intentionally keeps the task
fixed and comparable.

Target:

- Binary pCR (`pcr`)

Prediction settings:

- clinical-only baseline;
- image-only frozen foundation-model embeddings;
- image-plus-clinical fusion.

Probe protocol:

- frozen encoder;
- patient-level embedding aggregation;
- train-only standardization/imputation;
- L2-regularized logistic regression;
- hyperparameter selection inside official training data only;
- final evaluation on the official MAMA-MIA test set.

Primary metrics:

- AUROC;
- average precision;
- balanced accuracy.

## Benchmarking Pipeline

The first-pass benchmark is intentionally simple and leakage-safe. The goal is
to test whether frozen foundation-model representations contain useful pCR
signal before committing to full fine-tuning.

Pipeline:

1. Build the benchmark manifest from the official MAMA-MIA split file:
   `Benchmarking/outputs/manifests/mamamia_multiphase_foundation_manifest.csv`.
2. For each patient, select the requested imaging input: single DCE phase,
   post-contrast subtraction, raw phase fusion, subtraction fusion, all-DCE
   fusion, or Pillar-0 3D phase triplet.
3. Apply only model-specific preprocessing needed for inference, then pass the
   image input through a frozen foundation model.
4. Aggregate image features to one patient-level embedding row.
5. Train a lightweight L2-regularized logistic-regression probe for pCR.
6. Fit imputation and standardization on training data only.
7. Select `C` and the operating threshold using an inner validation split made
   only from official training patients.
8. Refit the final probe on all official training patients and evaluate once on
   the official MAMA-MIA test set.

This gives three directly comparable settings:

| Setting | Inputs to logistic regression | Purpose |
|---|---|---|
| Clinical-only | MAMA-MIA clinical/tabular variables | Establish a non-image baseline |
| Image-only | Frozen foundation-model patient embedding | Test image representation signal |
| Image + clinical | Patient embedding concatenated with clinical variables | Test whether imaging adds value beyond clinical data |

## Image Preprocessing Scope

The benchmark scripts perform model-specific preprocessing on the fly. This is
separate from the released dataset, whose NIfTI files remain in native
reconstructed geometry.

Examples:

- 2D foundation models sample slices from the source volumes, apply percentile
  clipping/min-max normalization, repeat grayscale slices to RGB, and resize to
  the model image size.
- Pillar-0 loads 3D phase volumes, optionally resamples to isotropic spacing,
  applies percentile clipping/min-max normalization, and center pads/crops to
  the model input shape.
- PCR classifier experiments load source NIfTI volumes, normalize crops, and
  resize/pad tensors during training.

These operations produce embeddings or tensors for a model run; they are not
written back into the public dataset release.

## Model Registry

The full live model table is stored in:

```text
Benchmarking/foundation_model_registry.csv
```

Models currently tracked:

| Priority | Model | Training modalities | Training scale | Disease/task context | Current role/status |
|---:|---|---|---|---|---|
| 1 | [Pillar-0 BreastMRI](https://huggingface.co/YalaLab/Pillar0-BreastMRI) | 3D breast MRI volumes paired with radiology reports | Not publicly specified in the model card | Breast MRI findings and report alignment | Primary 3D breast MRI baseline; whole-volume image-only done, ROI and image-plus-clinical pending |
| 2 | [MOME Breast mpMRI](https://www.nature.com/articles/s41467-025-58798-z) | Breast DCE-MRI, T2-weighted MRI, and DWI | 5220 MRI examinations from 5205 patients; NACT response subset n=358 | Breast malignancy diagnosis, TNBC subtyping, NACT response/pCR prediction | Candidate if usable pretrained weights/preprocessing are verified |
| 3 | [Curia](https://huggingface.co/raidium/curia) | CT and MRI cross-sectional slices | 150000 exams, 130 TB, more than 200M CT/MRI slices | Broad radiology tasks across anatomy, oncology, emergency, musculoskeletal, infectious, and neurodegenerative settings | First pass complete; strongest current image-only and balanced-accuracy multimodal result |
| 4 | [MedSigLIP](https://huggingface.co/google/medsiglip-448) | Medical image-text pairs including chest X-ray, dermatology, ophthalmology, histopathology, CT slices, MRI slices, plus natural image-text pairs | Not publicly specified in the model card | General medical image interpretation across multiple modalities | First pass complete as a broad 2D medical image-text baseline |
| 5 | [MedImageInsight](https://www.microsoft.com/en-us/research/publication/medimageinsight-an-open-source-embedding-model-for-general-domain-medical-imaging/) | X-ray, CT, MRI, dermoscopy, OCT, fundus photography, ultrasound, histopathology, and mammography | 3.7M clinical images from 14 medical domains | General-domain medical image classification, retrieval, and fine-tuning | Candidate if endpoint/local access is available |
| 6 | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) | Biomedical article figures and captions | 15M figure-caption pairs from PubMed Central | Broad biomedical literature figures, including but not limited to radiology | First pass complete; strongest current multimodal AUROC/AP |
| 7 | [RadImageNet](https://github.com/BMEII-AI/RadImageNet) | CT, MRI, and ultrasound radiology images | 1.35M annotated images from 131872 patients; 165 labels; 11 anatomic regions | Musculoskeletal, neurologic, oncologic, gastrointestinal, endocrine, abdominal, and pulmonary pathologies | Candidate CNN transfer baseline |
| 8 | [RadioDINO](https://github.com/Snarci/Radio-DINO) | Self-supervised ViT trained on RadImageNet CT, MRI, and ultrasound images | RadImageNet scale: 1.35M images | Radiomics and medical image analysis classification/segmentation tasks | First pass complete as a self-supervised radiology ViT baseline |
| 9 | [RadFM](https://github.com/chaoyi-wu/radfm) | 2D and 3D medical scans with text | MedMD: 16M 2D/3D images/scans, including about 15.5M 2D images and 500K 3D scans | General radiology vision-language tasks across 2D and 3D scans | Heavy stage-2 candidate for selected inputs only |
| 10 | [Jolia](https://huggingface.co/raidium/Jolia) | Adult chest and abdominal CT volumes with paired reports | 74434 public CT-report pairs from INSPECT, CT-RATE, and Stanford-Abdominal-CT | Chest/abdominal CT findings, per-organ concept alignment, and report-related tasks | Optional CT-only cross-modality stress test |

## Complete Experiment Matrix

The planned and completed benchmark ladder is tracked in:

```text
Benchmarking/experiment_matrix.md
```

The complete exported table of completed runs is:

```text
Benchmarking/outputs/summaries/cross_model_all_results.md
Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv
```

Current matrix snapshot:

| Category | Completed runs |
|---|---:|
| Clinical-only probes | 1 |
| Image-only embedding probes | 56 |
| Image-plus-clinical probes | 16 |
| Total completed probe runs | 73 |

The matrix varies four main experimental dimensions:

| Dimension | Values currently represented |
|---|---|
| Foundation model | Pillar-0 BreastMRI, RadioDINO, BiomedCLIP, Curia, MedSigLIP |
| Image crop | Whole volume, expert ROI |
| DCE input | Phase 0, phase 1, phase 2, last phase, post-contrast subtraction, raw phase fusion, subtraction fusion, all-DCE fusion, Pillar-0 3D phase triplets |
| Feature set | Clinical-only, image-only, image-plus-clinical |

## Current Results Snapshot

Snapshot date: 2026-09-15.

| Setting | Model | Crop/input | AUROC | AP | Bal Acc | Run |
|---|---|---|---:|---:|---:|---|
| Clinical-only | Clinical baseline | Clinical variables | 0.735 | 0.502 | 0.642 | `clinical_logreg` |
| Best image-only AUROC | Curia | Expert ROI, phase 2 - phase 0 | 0.630 | 0.434 | 0.578 | `curia_expert_roi_phase2_minus_phase0_logreg` |
| Best image-only AP | Curia | Expert ROI, selected ROI fusion | 0.608 | 0.446 | 0.546 | `curia_expert_roi_selected_fusion_logreg` |
| Best image + clinical AUROC/AP | BiomedCLIP | Whole volume, phase 1 + clinical | 0.739 | 0.553 | 0.605 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Best image + clinical balanced accuracy | Curia | Expert ROI, phase 2 - phase 0 + clinical | 0.724 | 0.543 | 0.658 | `curia_expert_roi_phase2_minus_phase0_image_clinical_logreg` |

The key interpretation so far is that clinical-only remains a strong baseline.
The best image-plus-clinical runs improve AP and/or balanced accuracy, but none
clearly dominates clinical-only across all metrics yet.

For detailed per-model tables with sensitivity, specificity, precision, F1,
embedding dimensionality, test-set size, and exact run names, see
[`docs/benchmark_results.md`](benchmark_results.md).

## Result Files

Headline summaries:

- `docs/benchmark_results.md`
- `Benchmarking/outputs/summaries/cross_model_first_pass_summary.md`
- `Benchmarking/outputs/summaries/cross_model_all_results.md`
- `Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv`
- `Benchmarking/outputs/summaries/cross_model_best_by_metric.csv`

Per-model summaries:

- `Benchmarking/outputs/summaries/radiodino_first_pass_summary.md`
- `Benchmarking/outputs/summaries/biomedclip_first_pass_summary.md`
- `Benchmarking/outputs/summaries/curia_first_pass_summary.md`
- `Benchmarking/outputs/summaries/medsiglip_first_pass_summary.md`
- `Benchmarking/outputs/summaries/pillar0_whole_volume_summary.md`

## Next Steps

- Complete Pillar-0 expert-ROI and image-plus-clinical checks.
- Add repeated seeds or cross-validation for shortlisted configurations.
- Extend the benchmark from original MAMA-MIA single-timepoint data to the
  reconstructed longitudinal release.
- Evaluate whether longitudinal features improve pCR prediction beyond
  pretreatment-only imaging and clinical variables.
