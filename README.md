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
- First-pass foundation-model pCR benchmark underway using frozen encoders,
  patient-level embedding aggregation, and L2 logistic-regression probes.
- Current benchmark snapshot date: 2026-09-15.

## Repository Layout

```text
.
|-- README.md
|-- docs/
|   |-- dataset.md
|   |-- reconstruction_workflow.md
|   |-- benchmarking.md
|   `-- repository_contents.md
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

See [docs/dataset.md](docs/dataset.md) for the dataset description, file tree,
metadata column groups, and citations.

## Reconstruction

The reconstruction work recovered additional longitudinal DCE-MRI timepoints for
MAMA-MIA patients whose source exams were available in ISPY1, ISPY2, and Breast
MRI NACT Pilot. The release builder creates clean public manifests and excludes
QC-quarantined patients from the active cohort.

See [docs/reconstruction_workflow.md](docs/reconstruction_workflow.md).

## Foundation-Model Benchmarking

The benchmark compares clinical-only, image-only, and image-plus-clinical pCR
prediction using frozen foundation-model embeddings. The model registry is in
[Benchmarking/foundation_model_registry.csv](Benchmarking/foundation_model_registry.csv)
and the live experiment matrix is in
[Benchmarking/experiment_matrix.md](Benchmarking/experiment_matrix.md).

Headline first-pass results so far:

| Setting | Best current result |
|---|---|
| Clinical-only baseline | AUROC 0.735, AP 0.502, balanced accuracy 0.642 |
| Image-only AUROC | Curia expert-ROI phase2-minus-phase0, AUROC 0.630 |
| Image-only AP | Curia expert-ROI selected ROI fusion, AP 0.446 |
| Image + clinical AUROC/AP | BiomedCLIP whole-volume phase1 + clinical, AUROC 0.739, AP 0.553 |
| Image + clinical balanced accuracy | Curia expert-ROI phase2-minus-phase0 + clinical, balanced accuracy 0.658 |

See [docs/benchmarking.md](docs/benchmarking.md) and
[Benchmarking/outputs/summaries/cross_model_first_pass_summary.md](Benchmarking/outputs/summaries/cross_model_first_pass_summary.md).

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
