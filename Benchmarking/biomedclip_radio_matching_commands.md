# BiomedCLIP RadioDINO-Matched Benchmark Commands

These commands complete BiomedCLIP using the same first-pass experiment ladder
used for RadioDINO.

Already done:

- `biomedclip_whole_phase1_logreg`
- `biomedclip_whole_phase2_minus_phase0_logreg`
- `biomedclip_expert_roi_phase2_minus_phase0_logreg`

## 1. Missing Whole-Volume Single Inputs

### Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_phase0_logreg
```

### Phase 2

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role phase2 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_phase2/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_phase2_logreg
```

### Last Phase

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role last_phase \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_last_phase/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_last_phase_logreg
```

## 2. Missing Whole-Volume Subtraction Inputs

### Phase 1 Minus Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role phase1_minus_phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_phase1_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_phase1_minus_phase0_logreg
```

### Last Phase Minus Phase 0

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role last_phase_minus_phase0 \
  --crop-mode whole \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_last_phase_minus_phase0_logreg
```

## 3. Whole-Volume Fusion

### Raw Phase Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/biomedclip/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/biomedclip/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/biomedclip/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/biomedclip/whole_last_phase/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/biomedclip/whole_raw_phases_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_raw_phases_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_raw_phases_fusion_logreg
```

### Subtraction Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input p1sub=Benchmarking/outputs/embeddings/biomedclip/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/biomedclip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/biomedclip/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/biomedclip/whole_subtractions_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_subtractions_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_subtractions_fusion_logreg
```

### All DCE Fusion

```bash
./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/biomedclip/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/biomedclip/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/biomedclip/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/biomedclip/whole_last_phase/patient_embeddings.csv \
  --input p1sub=Benchmarking/outputs/embeddings/biomedclip/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/biomedclip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/biomedclip/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/biomedclip/whole_all_dce_fusion/patient_embeddings.csv
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_all_dce_fusion_logreg
```

## 4. Missing ROI Input

BiomedCLIP whole phase1 is currently its best image-only result, so test phase1
ROI too.

```bash
./.venv/bin/python Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key biomedclip \
  --input-role phase1 \
  --crop-mode expert_roi \
  --max-slices 16 \
  --batch-size 16 \
  --device auto
```

```bash
./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/expert_roi_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_expert_roi_phase1_logreg
```

## 5. Image Plus Clinical

These mirror the RadioDINO multimodal checks.

### Whole Phase 1 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_phase1_image_clinical_logreg
```

### Whole Phase 2 Minus Phase 0 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_phase2_minus_phase0_image_clinical_logreg
```

### Whole All DCE Fusion Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_whole_all_dce_fusion_image_clinical_logreg
```

### Expert ROI Phase 2 Minus Phase 0 Plus Clinical

```bash
./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/biomedclip/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/biomedclip_expert_roi_phase2_minus_phase0_image_clinical_logreg
```

## 6. Summary

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
```
