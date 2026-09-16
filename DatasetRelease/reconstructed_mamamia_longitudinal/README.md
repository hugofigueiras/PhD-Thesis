# Reconstructed MAMA-MIA Longitudinal Dataset

This local release package documents the reconstructed longitudinal DCE-MRI
volumes derived from ISPY1, ISPY2, and Breast MRI NACT Pilot patients that
also appear in MAMA-MIA. The NIfTI images are not copied into this folder;
the manifests point to the active local image folders with project-relative
paths.

## Active Image Roots

- `Reconstructed_Datasets/ISPY1-Volumes/`: 145 patients, 549 exams, 1705 phase files
- `Reconstructed_Datasets/ISPY2-Volumes/`: 716 patients, 2680 exams, 19159 phase files
- `Reconstructed_Datasets/NACT-Volumes/`: 64 patients, 189 exams, 588 phase files

Folder convention:

```text
Reconstructed_Datasets/<dataset-volume-root>/<reconstructed-patient-id>/<MM-DD-YYYY>/volume_phase<index>.nii.gz
```

Public-facing `patient_id` values follow MAMA-MIA naming. ISPY2 image
folders use hyphens locally, for example `ISPY2-100899`, while MAMA-MIA
tables use `ISPY2_100899`; both identifiers are recorded in the manifests.

## Release Cohort

- Active reconstructed patients: 925
- Active reconstructed exams/timepoints: 3418
- Active reconstructed NIfTI phase files: 21452
- Quarantined patients excluded from active release: 12

MAMA-MIA status counts:

- `active_reconstructed`: 925
- `quarantined`: 12
- `no_active_reconstructed_volume`: 278
- `not_reconstructed_source_dataset`: 291

Use `metadata/clinical_and_imaging_info_active_reconstructed.csv` as the
cleaned MAMA-MIA-style tabular table for this reconstructed dataset. The
full status table is kept as an audit trail.

## Metadata Files

- `metadata/patient_manifest.csv`: active patient-level image inventory and patient-level QC summaries.
- `metadata/exam_manifest.csv`: active exam/timepoint-level inventory and reconstruction audit fields.
- `metadata/phase_manifest.csv`: active phase-level NIfTI path list.
- `metadata/quarantine_manifest.csv`: excluded quarantined patients and reasons.
- `metadata/clinical_and_imaging_info_active_reconstructed.csv`: MAMA-MIA rows filtered to active reconstructed patients.
- `metadata/clinical_and_imaging_info_with_reconstruction_status.csv`: full MAMA-MIA table with release status.
- `metadata/clinical_and_imaging_info_nonreleased.csv`: MAMA-MIA rows outside the active reconstructed release.
- `metadata/clinical_and_imaging_info_active_reconstructed.xlsx`: workbook copy with active, full-status, nonreleased, and legend sheets.
- `metadata/release_summary.json`: machine-readable counts and provenance.
- `metadata/data_dictionary.csv`: short description of each generated metadata file.

## Quarantine Policy

Patients in `quarantine_manifest.csv` are excluded from the active release.
Current quarantine sources are geometry failures and view mismatch cases
under `Reconstructed_Datasets/Quarantine/`. The original quarantined files
remain on disk for audit/debugging but are not part of the active dataset.

## Regeneration

Run from the project root:

```bash
./.venv/bin/python ReconstructionScripts/build_reconstructed_dataset_release.py
```

Inputs used for this generated package:

- Reconstructed root: `Reconstructed_Datasets`
- MAMA-MIA clinical table: `Original_Datasets/MAMA-MIA/clinical_and_imaging_info.xlsx`
- Output root: `DatasetRelease/reconstructed_mamamia_longitudinal`

Large NIfTI files should stay out of ordinary Git history. For a public
release, keep this metadata/docs package in GitHub and place images on a
data host such as Hugging Face Datasets, Git LFS, DVC, or an institutional
archive.
