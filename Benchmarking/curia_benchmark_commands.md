# Curia Foundation-Model Benchmark Commands

Curia is the next selected foundation model after RadioDINO and BiomedCLIP.
It is a radiology foundation model loaded with Hugging Face Transformers.

Before running, make sure you have accepted the Curia model terms on Hugging
Face and that your local environment is logged in using the same cache location
as the extractor:

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/hf auth login
```

Verify the active account without printing the token:

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

Already done:

- Curia smoke test
- `curia_whole_phase0_logreg`
- `curia_whole_phase1_logreg`
- `curia_whole_phase2_logreg`
- `curia_whole_last_phase_logreg`
- `curia_whole_phase1_minus_phase0_logreg`
- `curia_whole_phase2_minus_phase0_logreg`
- `curia_whole_last_phase_minus_phase0_logreg`
- `curia_whole_raw_phases_fusion_logreg`
- `curia_whole_subtractions_fusion_logreg`
- `curia_whole_all_dce_fusion_logreg`
- `curia_expert_roi_phase1_logreg`
- `curia_expert_roi_phase2_minus_phase0_logreg`
- `curia_expert_roi_selected_fusion_logreg`
- `curia_whole_phase1_image_clinical_logreg`
- `curia_whole_phase2_minus_phase0_image_clinical_logreg`
- `curia_whole_all_dce_fusion_image_clinical_logreg`
- `curia_expert_roi_phase2_minus_phase0_image_clinical_logreg`

Curia first-pass benchmark is complete.

Run next:

- choose the next foundation model from the registry, or
- run stronger validation on the current shortlist

## 0. Smoke Test

Run this first. It checks access, model loading, preprocessing, embedding shape,
and output writing without doing the full dataset.

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase1 \
  --crop-mode whole \
  --max-patients 8 \
  --max-slices 8 \
  --batch-size 4 \
  --device auto
```

If this succeeds, continue below. The full phase1 extraction will overwrite the
8-patient smoke output.

## 1. Whole-Volume Single Inputs

### Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase0_logreg
```

### Phase 1

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase1 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase1_logreg
```

### Phase 2

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase2 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase2/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase2_logreg
```

### Last Phase

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role last_phase \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_last_phase/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_last_phase_logreg
```

## 2. Whole-Volume Subtraction Inputs

### Phase 1 Minus Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase1_minus_phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase1_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase1_minus_phase0_logreg
```

### Phase 2 Minus Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase2_minus_phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase2_minus_phase0_logreg
```

### Last Phase Minus Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role last_phase_minus_phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_last_phase_minus_phase0_logreg
```

## 3. Whole-Volume Fusion

### Raw Phase Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/curia/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/curia/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/curia/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/curia/whole_last_phase/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/curia/whole_raw_phases_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_raw_phases_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_raw_phases_fusion_logreg
```

### Subtraction Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input p1sub=Benchmarking/outputs/embeddings/curia/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/curia/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/curia/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/curia/whole_subtractions_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_subtractions_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_subtractions_fusion_logreg
```

### All DCE Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/curia/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/curia/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/curia/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/curia/whole_last_phase/patient_embeddings.csv \
  --input p1sub=Benchmarking/outputs/embeddings/curia/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/curia/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/curia/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/curia/whole_all_dce_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_all_dce_fusion_logreg
```

## 4. Expert ROI Inputs

### Phase 1 ROI

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase1 \
  --crop-mode expert_roi \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/expert_roi_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_expert_roi_phase1_logreg
```

### Phase 2 Minus Phase 0 ROI

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key curia \
  --input-role phase2_minus_phase0 \
  --crop-mode expert_roi \
  --max-slices 16 \
  --batch-size 4 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_expert_roi_phase2_minus_phase0_logreg
```

### Selected ROI Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase1=Benchmarking/outputs/embeddings/curia/expert_roi_phase1/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/curia/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/curia/expert_roi_selected_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/expert_roi_selected_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_expert_roi_selected_fusion_logreg
```

## 5. Image Plus Clinical

### Whole Phase 1 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase1_image_clinical_logreg
```

### Whole Phase 2 Minus Phase 0 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_phase2_minus_phase0_image_clinical_logreg
```

### Whole All DCE Fusion Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_whole_all_dce_fusion_image_clinical_logreg
```

### Expert ROI Phase 2 Minus Phase 0 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/curia/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/curia_expert_roi_phase2_minus_phase0_image_clinical_logreg
```

## 6. Summary

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
```
