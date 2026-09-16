# MedSigLIP Foundation-Model Benchmark Commands

MedSigLIP is the next selected 2D slice foundation model after Curia. It is a
448 px medical image-text SigLIP model from Google. It is gated on Hugging Face,
so accept the model terms before running if you have not already.

These commands use the same first-pass benchmark ladder as RadioDINO,
BiomedCLIP, and Curia:

- whole-volume single phases
- whole-volume subtraction inputs
- whole-volume feature fusion
- expert ROI single inputs
- expert ROI selected fusion
- image-plus-clinical fusion probes

Operational note: MedSigLIP is heavier than BiomedCLIP/RadioDINO, so the default
batch size here is 1. This changes memory use, not the benchmark definition. The
commands also write extraction logs under `Benchmarking/outputs/logs/` so a long
run does not overwhelm the terminal scrollback.

## 0. Login And Smoke Test

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/hf auth login
```

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

Run this first. It checks access, model loading, image preprocessing, embedding
shape, and output writing without running the full dataset. The first smoke is
intentionally tiny and CPU-only, which makes it the safest way to diagnose model
loading without crashing the terminal.

```bash
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

./.venv/bin/python -u Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key medsiglip \
  --input-role phase1 \
  --crop-mode whole \
  --max-patients 1 \
  --max-slices 1 \
  --batch-size 1 \
  --device cpu \
  --torch-dtype float32 \
  --log-every 1 \
  > Benchmarking/outputs/logs/medsiglip_cpu_tiny_smoke.log 2>&1

status=$?
tail -80 Benchmarking/outputs/logs/medsiglip_cpu_tiny_smoke.log
printf "\nMedSigLIP CPU smoke exit status: %s\n" "$status"
```

If that works, run the slightly larger auto-device smoke before the full ladder:

```bash
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

./.venv/bin/python -u Benchmarking/extract_2d_foundation_embeddings.py \
  --model-key medsiglip \
  --input-role phase1 \
  --crop-mode whole \
  --max-patients 4 \
  --max-slices 4 \
  --batch-size 1 \
  --device auto \
  --torch-dtype auto \
  --log-every 1 \
  > Benchmarking/outputs/logs/medsiglip_auto_smoke.log 2>&1

status=$?
tail -80 Benchmarking/outputs/logs/medsiglip_auto_smoke.log
printf "\nMedSigLIP auto smoke exit status: %s\n" "$status"
```

The full phase1 command below will overwrite the 4-patient smoke output.

## 1. Whole-Volume Single And Subtraction Inputs

```bash
set -euo pipefail
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

for role in phase0 phase1 phase2 last_phase phase1_minus_phase0 phase2_minus_phase0 last_phase_minus_phase0; do
  ./.venv/bin/python -u Benchmarking/extract_2d_foundation_embeddings.py \
    --model-key medsiglip \
    --input-role "$role" \
    --crop-mode whole \
    --max-slices 16 \
    --batch-size 1 \
    --device auto \
    --torch-dtype auto \
    --log-every 25 \
    > "Benchmarking/outputs/logs/medsiglip_whole_${role}_extract.log" 2>&1

  ./.venv/bin/python Benchmarking/train_embedding_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/medsiglip/whole_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag usable_for_whole_volume_benchmark \
    --output-dir "Benchmarking/outputs/probes/medsiglip_whole_${role}_logreg"
done
```

## 2. Whole-Volume Fusion

```bash
set -euo pipefail

./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/medsiglip/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/medsiglip/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/medsiglip/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/medsiglip/whole_last_phase/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/medsiglip/whole_raw_phases_fusion/patient_embeddings.csv

./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_raw_phases_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_raw_phases_fusion_logreg

./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input p1sub=Benchmarking/outputs/embeddings/medsiglip/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/medsiglip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/medsiglip/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/medsiglip/whole_subtractions_fusion/patient_embeddings.csv

./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_subtractions_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_subtractions_fusion_logreg

./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase0=Benchmarking/outputs/embeddings/medsiglip/whole_phase0/patient_embeddings.csv \
  --input phase1=Benchmarking/outputs/embeddings/medsiglip/whole_phase1/patient_embeddings.csv \
  --input phase2=Benchmarking/outputs/embeddings/medsiglip/whole_phase2/patient_embeddings.csv \
  --input last=Benchmarking/outputs/embeddings/medsiglip/whole_last_phase/patient_embeddings.csv \
  --input p1sub=Benchmarking/outputs/embeddings/medsiglip/whole_phase1_minus_phase0/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/medsiglip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --input lastsub=Benchmarking/outputs/embeddings/medsiglip/whole_last_phase_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/medsiglip/whole_all_dce_fusion/patient_embeddings.csv

./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_all_dce_fusion_logreg
```

## 3. Expert ROI Inputs

```bash
set -euo pipefail
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

for role in phase1 phase2_minus_phase0; do
  ./.venv/bin/python -u Benchmarking/extract_2d_foundation_embeddings.py \
    --model-key medsiglip \
    --input-role "$role" \
    --crop-mode expert_roi \
    --max-slices 16 \
    --batch-size 1 \
    --device auto \
    --torch-dtype auto \
    --log-every 25 \
    > "Benchmarking/outputs/logs/medsiglip_expert_roi_${role}_extract.log" 2>&1

  ./.venv/bin/python Benchmarking/train_embedding_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/medsiglip/expert_roi_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag usable_for_expert_roi_benchmark \
    --output-dir "Benchmarking/outputs/probes/medsiglip_expert_roi_${role}_logreg"
done

./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase1=Benchmarking/outputs/embeddings/medsiglip/expert_roi_phase1/patient_embeddings.csv \
  --input p2sub=Benchmarking/outputs/embeddings/medsiglip/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/medsiglip/expert_roi_selected_fusion/patient_embeddings.csv

./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/expert_roi_selected_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_expert_roi_selected_fusion_logreg
```

## 4. Image Plus Clinical

```bash
set -euo pipefail

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_phase1/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_phase1_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_phase2_minus_phase0_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/whole_all_dce_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_whole_volume_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_whole_all_dce_fusion_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/medsiglip/expert_roi_phase2_minus_phase0/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/medsiglip_expert_roi_phase2_minus_phase0_image_clinical_logreg
```

## 5. Summary

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
```
