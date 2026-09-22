# PhD Thesis: Longitudinal Multimodal Breast MRI pCR Prediction

This repository tracks the code, documentation, dataset-release workflow, and
foundation-model benchmarks for my PhD project on pathological complete response
(pCR) prediction from breast DCE-MRI and tabular clinical data.

The current public data artifact is hosted on Hugging Face:

- Dataset: https://huggingface.co/datasets/hfigueiras/LMBM

GitHub is used for the research workflow: reconstruction scripts, release
documentation, foundation-model benchmark scripts, model registry, and current
benchmark result summaries. Large NIfTI volumes, model checkpoints, embeddings,
local caches, and original DICOM/source-data mirrors are intentionally excluded.

## Current Status

- Reconstructed longitudinal MAMA-MIA-derived DCE-MRI dataset released on
  Hugging Face.
- Active release cohort: 925 patients, 3418 MRI exams/timepoints, and 21452
  DCE-MRI NIfTI phase files.
- Public metadata table:
  `clinical_and_imaging_info_active_reconstructed.csv`, with 58 public columns.
- Feasible first-pass foundation-model pCR registry screen complete using frozen
  encoders, patient-level embeddings, and L2 logistic-regression probes.
- Current benchmark matrix: 106 completed probe runs, including 98 primary
  modality-matched/reference runs and 8 Jolia CT-to-MRI stress-test runs.
- Aggregate paired-bootstrap and source-cohort robustness analyses complete.
- Current benchmark snapshot date: 2026-09-22.

## Repository Layout

```text
.
|-- README.md
|-- docs/
|   |-- dataset.md
|   |-- reconstruction_workflow.md
|   |-- benchmarking.md
|   |-- benchmark_results.md
|   `-- repository_contents.md
|-- presentations/
|   `-- advisor_update_2026-09-16.md
|-- DatasetRelease/
|   `-- reconstructed_mamamia_longitudinal/
|       |-- README.md
|       |-- prepare_hf_upload.py
|       `-- metadata/
|           |-- data_dictionary.csv
|           `-- release_summary.json
|-- ReconstructionScripts/
|   |-- build_reconstructed_dataset_release.py
|   |-- audit_*.py
|   |-- make-volumes*.py
|   `-- qc_reports/*/summary.json
|-- Benchmarking/
|   |-- foundation_model_registry.csv
|   |-- experiment_matrix.md
|   |-- *.py
|   `-- outputs/summaries/
`-- PCRPredictionScripts/
    |-- *.py
    |-- README.md
    `-- outputs/*.csv
```

## Dataset

The dataset is a multimodal, longitudinal breast cancer response dataset. It
combines:

- volumetric DCE-MRI images stored as NIfTI phase files;
- patient-level clinical, treatment, tumor-biology, outcome, demographic, body
  habitus, scanner, and acquisition metadata;
- longitudinal reconstruction/linkage metadata showing which patients have
  multiple recovered MRI exams/timepoints.

The released NIfTI images are kept in their native reconstructed geometry. They
are not globally resampled, resized, reoriented, or intensity-normalized as a
dataset-level preprocessing step. Model-specific resizing, resampling,
normalization, cropping, and padding are applied later on the fly inside the
benchmark/training pipelines.

See [docs/dataset.md](docs/dataset.md) for the dataset description, file tree,
metadata column groups, and citations.

## Advisor Update Deck

A Markdown slide deck for the current project update is available at
[presentations/advisor_update_2026-09-16.md](presentations/advisor_update_2026-09-16.md).

## Reconstruction

The reconstruction work recovered additional longitudinal DCE-MRI timepoints for
MAMA-MIA patients whose source exams were available in ISPY1, ISPY2, and Breast
MRI NACT Pilot. The release builder creates clean public manifests and excludes
QC-quarantined patients from the active cohort.

See [docs/reconstruction_workflow.md](docs/reconstruction_workflow.md).

## Foundation-Model Benchmarking

The benchmark compares clinical-only, image-only, and image-plus-clinical pCR
prediction using frozen foundation-model embeddings. It uses the official
MAMA-MIA train/test split, extracts one patient-level embedding per model/input,
and then trains an L2-regularized logistic-regression probe for pCR
classification. Clinical fusion runs concatenate the embedding with the
leakage-safe clinical variables before fitting the same probe.

The model registry is in
[Benchmarking/foundation_model_registry.csv](Benchmarking/foundation_model_registry.csv)
and the live experiment matrix is in
[Benchmarking/experiment_matrix.md](Benchmarking/experiment_matrix.md).
The full completed-run table is in
[Benchmarking/outputs/summaries/cross_model_all_results.md](Benchmarking/outputs/summaries/cross_model_all_results.md).

Benchmark preprocessing is model-specific. For example, 2D foundation models
resize sampled slices to their required image size, while Pillar-0 resamples
3D inputs to isotropic spacing before center pad/crop. These operations do not
modify the released dataset files.

Headline first-pass results so far:

| Setting | Model | Crop/input | AUROC | AP | Bal Acc | Run |
|---|---|---|---:|---:|---:|---|
| Clinical-only | Clinical baseline | Clinical variables | 0.735 | 0.502 | 0.642 | `clinical_logreg` |
| Best image-only AUROC | Curia | Expert ROI, phase 2 - phase 0 | 0.630 | 0.434 | 0.578 | `curia_expert_roi_phase2_minus_phase0_logreg` |
| Best image-only AP | Curia | Expert ROI, selected ROI fusion | 0.608 | 0.446 | 0.546 | `curia_expert_roi_selected_fusion_logreg` |
| Best image-only balanced accuracy | RadImageNet | Whole volume, phase 2 - phase 0 | 0.619 | 0.396 | 0.593 | `radimagenet_whole_phase2_minus_phase0_logreg` |
| Best image + clinical AUROC/AP | BiomedCLIP | Whole volume, phase 1 + clinical | 0.739 | 0.553 | 0.605 | `biomedclip_whole_phase1_image_clinical_logreg` |
| Best image + clinical balanced accuracy | RadImageNet | Whole volume, phase 2 - phase 0 + clinical | 0.725 | 0.510 | 0.681 | `radimagenet_whole_phase2_minus_phase0_image_clinical_logreg` |

Despite those point-estimate winners, a 5,000-resample paired bootstrap found no
primary image-plus-clinical configuration with a stable aggregate improvement
over clinical-only. The main follow-up hypothesis is cohort-specific:
BiomedCLIP plus clinical improves AUROC and AP within the ISPY2 test subgroup,
but this requires source-held-out validation before a generalization claim.

See [docs/benchmarking.md](docs/benchmarking.md) and
[docs/benchmark_results.md](docs/benchmark_results.md) for the detailed
per-model result tables. The full exported table is also stored in
[Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv](Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv).

## What Is Not Stored Here

This repository does not store:

- original DICOM files;
- reconstructed NIfTI image volumes;
- Hugging Face upload staging hardlinks;
- generated embeddings;
- trained checkpoints and probe model files;
- W&B caches, Hugging Face caches, tokens, or local environment files;
- copyrighted paper PDFs.

Those exclusions keep the repository readable, public-safe, and updateable as
new benchmark results are generated.
