# Foundation-Model Benchmarking

The benchmark asks whether frozen medical foundation-model representations
contain useful signal for pathological complete response (pCR) prediction and
whether they add value beyond tabular clinical variables.

The feasible first-pass registry screen was completed on 2026-09-22. Detailed
scores and uncertainty analyses are reported in
[`benchmark_results.md`](benchmark_results.md).

## Benchmark Dataset

The benchmark uses the original MAMA-MIA pretreatment DCE-MRI exam and official
train/test split. Reconstructed longitudinal timepoints are intentionally held
for a later experiment so that all foundation models are first compared on the
same single-exam task.

| Field | Value |
|---|---|
| Target | Binary pCR |
| Official training patients | 1,185 |
| Official test patients | 306 |
| Test pCR cases | 93 |
| Primary metrics | AUROC, average precision, balanced accuracy |

Three feature settings are compared:

| Setting | Probe inputs | Question |
|---|---|---|
| Clinical-only | Leakage-safe MAMA-MIA clinical variables | How strong is the non-image reference? |
| Image-only | Frozen patient-level image embedding | Does the representation contain pCR signal? |
| Image + clinical | Image embedding concatenated with clinical variables | Does imaging add value beyond clinical data? |

## Benchmark Pipeline

1. Build a patient manifest from the official split and available DCE phases.
2. Select a single phase, subtraction, phase fusion, or expert-ROI input.
3. Apply the preprocessing required by the frozen foundation model.
4. Aggregate slice features to one patient vector for 2D models, or extract one
   volume-level vector for 3D models.
5. Fit an L2-regularized logistic-regression probe.
6. Fit imputation and standardization using official training patients only.
7. Select regularization strength and the operating threshold on an inner
   validation split made from official training patients.
8. Refit on all official training patients and evaluate on the official test
   patients.

Within-run fitting is leakage-safe. Across runs, however, the model/input
shortlist was chosen after comparing many configurations on the same test set.
Leaderboard winners and bootstrap intervals are therefore exploratory rather
than confirmatory.

## Imaging Inputs

For 2D encoders, slices are sampled from the 3D volume, intensity-normalized,
converted to model-compatible RGB inputs, encoded independently, and averaged
to one patient embedding. Feature-level phase fusion concatenates patient
embeddings after the frozen encoder.

For Pillar-0, three 3D volumes are stacked as channels before the native 3D
encoder. The tested triplets contain raw DCE phases or post-contrast
subtractions.

Expert-ROI crops use the MAMA-MIA expert mask on phase 0. The tumor bounding box
is expanded by 30 x 30 x 20 mm, clipped to image bounds, and applied at the same
coordinates to all phases and subtractions. Crops are generated in memory; the
source NIfTI files are unchanged.

## Completed Registry Screen

The complete registry is stored at
[`Benchmarking/foundation_model_registry.csv`](../Benchmarking/foundation_model_registry.csv).

| Model | Training alignment | Benchmark status |
|---|---|---|
| Pillar-0 BreastMRI | Native 3D breast MRI | Complete |
| RadioDINO | Self-supervised CT/MRI/ultrasound radiology images | Complete |
| BiomedCLIP | Biomedical image-text pairs | Complete |
| Curia | CT/MRI radiology slices | Complete |
| MedSigLIP | Broad medical image-text pairs | Complete |
| RadImageNet | Supervised radiology image classification | Complete |
| Jolia | Chest/abdominal CT volumes | Complete as a separate negative transfer control |
| MOME Breast mpMRI | Multiparametric breast MRI | Excluded because required DWI/T2 inputs are unavailable |
| MedImageInsight | General medical image embeddings | Access blocked by Azure Limited Preview requirements |
| RadFM | Large 2D/3D radiology VLM | Resource blocked on current hardware/software path |

## Experiment Matrix

The live and completed experiment table is in
[`Benchmarking/experiment_matrix.md`](../Benchmarking/experiment_matrix.md).

| Category | Completed probes |
|---|---:|
| Clinical-only | 1 |
| Image-only | 77 |
| Image + clinical | 28 |
| **Total** | **106** |

The total contains 98 primary modality-matched/reference runs and 8 Jolia
CT-to-MRI stress-test runs.

## Headline Results

| Setting | Model/input | AUROC | AP | Bal Acc |
|---|---|---:|---:|---:|
| Clinical-only | Clinical variables | 0.735 | 0.502 | 0.642 |
| Best image-only AUROC | Curia ROI, phase 2 - phase 0 | 0.630 | 0.434 | 0.578 |
| Best image-only AP | Curia ROI, selected fusion | 0.608 | 0.446 | 0.546 |
| Best image-only Bal Acc | RadImageNet whole, phase 2 - phase 0 | 0.619 | 0.396 | 0.593 |
| Best multimodal AUROC/AP | BiomedCLIP whole phase 1 + clinical | 0.739 | 0.553 | 0.605 |
| Best multimodal Bal Acc | RadImageNet whole subtraction + clinical | 0.725 | 0.510 | 0.681 |

A 5,000-resample paired bootstrap found no primary image-plus-clinical
configuration with a stable aggregate improvement over clinical-only on AUROC,
AP, or balanced accuracy. The strongest cohort-specific result occurs in ISPY2:
BiomedCLIP plus clinical improves AUROC by +0.099 [0.026, 0.177] and AP by
+0.188 [0.073, 0.267]. This is an exploratory subgroup result, not external
validation.

## Interpretation

- Frozen foundation-model embeddings contain modest image-only pCR signal.
- No image-only configuration outperforms the clinical baseline.
- Adding image embeddings to clinical variables changes point estimates but
  does not show a reliable aggregate benefit.
- ROI cropping is model-dependent: it helps Curia and MedSigLIP, while
  BiomedCLIP and RadImageNet favor whole-volume inputs.
- Single phases or subtractions often perform as well as, or better than,
  larger concatenated feature sets.
- Source-cohort heterogeneity is substantial and should shape the next
  validation design.

## Reproducing The Summaries

Existing probe outputs can be summarized without rerunning image extraction:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
./.venv/bin/python Benchmarking/bootstrap_benchmark_shortlist.py --n-bootstrap 5000 --seed 20260921
./.venv/bin/python Benchmarking/analyze_shortlist_by_dataset.py --n-bootstrap 5000 --seed 20260922
```

Important output files:

- `Benchmarking/outputs/summaries/probe_run_summary.csv`
- `Benchmarking/outputs/summaries/cross_model_all_results.md`
- `Benchmarking/outputs/summaries/cross_model_best_by_metric.csv`
- `Benchmarking/outputs/summaries/bootstrap_shortlist_summary.md`
- `Benchmarking/outputs/summaries/shortlist_by_dataset_summary.md`
- `Benchmarking/outputs/summaries/jolia_cross_modality_summary.md`

## Next Experimental Stage

The current test set should now be locked against further configuration
selection. The next study should predeclare a small shortlist and evaluate
source generalization, preferably with leave-one-source-dataset-out training.
BiomedCLIP is the primary frozen candidate because of its ISPY2 result. A
supervised CNN/ViT baseline should then determine whether frozen embeddings are
preferable to task-specific learning before longitudinal modeling begins.
