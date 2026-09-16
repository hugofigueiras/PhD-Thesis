# Dataset: LMBM Reconstructed MAMA-MIA Longitudinal DCE-MRI

Dataset location:

- Hugging Face: https://huggingface.co/datasets/hfigueiras/LMBM

This dataset contains reconstructed longitudinal breast DCE-MRI volumes and
patient-level tabular metadata for MAMA-MIA patients whose source data came from
ISPY1, ISPY2, and Breast MRI NACT Pilot. Patients excluded during local QC are
not included in the active public upload.

## Dataset Description

This is a multimodal, longitudinal breast cancer response dataset.

The imaging modality is volumetric DCE-MRI stored as NIfTI phase files
(`volume_phase*.nii.gz`). The tabular modality contains patient-level clinical,
demographic/body-habitus, treatment, tumor biology, outcome, scanner, and
acquisition metadata inherited from MAMA-MIA, plus reconstruction fields added
for this longitudinal release.

The main response label inherited from MAMA-MIA is `pcr`, pathological complete
response to neoadjuvant chemotherapy. Longitudinal structure is represented by
multiple reconstructed MRI exams/timepoints per patient when the source TCIA
data contained additional exams.

## Contents

- Active patients: 925
- Active exams/timepoints: 3418
- Active NIfTI phase files: 21452
- Source cohorts: ISPY1, ISPY2, and Breast MRI NACT Pilot
- Timepoints per patient: 15 patients have 1 exam/timepoint, 52 patients have 2
  exams/timepoints, 133 patients have 3 exams/timepoints, and 725 patients have
  4 exams/timepoints.
- DCE phase files per exam: 1-11

## Image Geometry And Preprocessing

The released NIfTI files are stored in their native reconstructed geometry. The
dataset release does **not** apply a single global preprocessing grid, such as a
shared voxel spacing, matrix size, orientation, or intensity normalization.

The reconstruction scripts convert DICOM phase stacks to NIfTI and preserve the
DICOM-derived spacing, origin, and direction in the image header. QC then checks
that phases within the same exam are geometrically consistent and excludes
quarantined cases from the active release.

Current active-release geometry summary:

| Property | Value |
|---|---:|
| Unique reconstructed shapes | 130 |
| Unique reconstructed voxel spacings | 287 |
| Axial exams/timepoints | 2680 |
| Sagittal exams/timepoints | 738 |
| Exams with phase-consistent shape | 3418 / 3418 |
| Exams with phase-consistent spacing | 3418 / 3418 |
| Exams with phase-consistent affine | 3418 / 3418 |

Model-specific preprocessing is applied later inside benchmark/training
pipelines. For example, 2D foundation-model extraction normalizes and resizes
sampled slices on the fly, and the Pillar-0 3D extraction pipeline resamples,
normalizes, and center pads/crops inputs before model inference. Those operations
do not overwrite or modify the released NIfTI files.

## Dataset Organization

```text
.
|-- README.md
|-- metadata/
|   |-- clinical_and_imaging_info_active_reconstructed.csv
|   |-- patient_manifest.csv
|   |-- exam_manifest.csv
|   |-- phase_manifest.csv
|   `-- data_dictionary.csv
`-- Reconstructed_Datasets/
    |-- ISPY1-Volumes/                         # 145 patients, 549 exams, 1705 phase files
    |   `-- ISPY1_1001/
    |       `-- 10-13-1984/
    |           |-- volume_phase0.nii.gz
    |           |-- volume_phase1.nii.gz
    |           `-- volume_phase2.nii.gz
    |-- ISPY2-Volumes/                         # 716 patients, 2680 exams, 19159 phase files
    |   `-- ISPY2-100899/
    |       `-- 10-26-2002/
    |           |-- volume_phase0.nii.gz
    |           |-- volume_phase1.nii.gz
    |           `-- ...
    `-- NACT-Volumes/                          # 64 patients, 189 exams, 588 phase files
        `-- NACT_01/
            `-- <MM-DD-YYYY>/
                |-- volume_phase0.nii.gz
                |-- volume_phase1.nii.gz
                `-- ...
```

Public-facing `patient_id` values in the metadata follow MAMA-MIA naming. ISPY2
folders use hyphens locally, for example `ISPY2-100899`; the metadata also
records the MAMA-MIA-style identifier `ISPY2_100899`.

## Clinical And Imaging Metadata

`metadata/clinical_and_imaging_info_active_reconstructed.csv` contains one row
per active released patient. It has 58 public columns: the 50 MAMA-MIA
clinical/imaging fields for the active reconstructed cohort plus 8
reconstruction/linkage fields added for this longitudinal release.

| Column group | Count |
|---|---:|
| Identifier/source-provenance variables | 2 |
| Clinical, treatment, outcome, and tumor-biology variables | 19 |
| Demographic/body-habitus variables | 8 |
| Imaging/acquisition/scanner/TCIA-series variables | 21 |
| Reconstruction and longitudinal-linkage variables | 8 |
| Total public columns | 58 |

## Source Datasets

This release is derived from public TCIA/MAMA-MIA source data. Users should cite
MAMA-MIA, the original source datasets represented here, and TCIA.

- MAMA-MIA: Garrucho et al., Scientific Data, 2025,
  DOI `10.1038/s41597-025-04707-4`
- ISPY1: DOI `10.7937/K9/TCIA.2016.HdHpgJLK`
- Breast MRI NACT Pilot: DOI `10.7937/K9/TCIA.2016.QHsyhJKy`
- ISPY2: DOI `10.7937/TCIA.D8Z0-9T85`
- TCIA: Clark et al., Journal of Digital Imaging, 2013,
  DOI `10.1007/s10278-013-9622-7`

## Use Restrictions

Do not attempt to identify or contact individual participants. Follow TCIA and
MAMA-MIA data usage policies and attribution requirements. The Hugging Face
dataset card specifies the current non-commercial Creative Commons license.
