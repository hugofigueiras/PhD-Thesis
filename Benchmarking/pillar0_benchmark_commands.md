# Pillar-0 BreastMRI Benchmark Commands

Pillar-0 BreastMRI is the next benchmark target after RadioDINO, BiomedCLIP,
Curia, and MedSigLIP. Unlike those 2D slice encoders, Pillar-0 is a 3D breast
MRI foundation model. It should therefore be benchmarked with 3-channel DCE-MRI
volumes, not with independent sampled slices.

The extractor writes patient-level embeddings to:

```text
Benchmarking/outputs/embeddings/pillar0_breastmri/<crop>_<input_role>/patient_embeddings.csv
```

The first-pass Pillar-0 input roles are:

| Input role | Channels |
|---|---|
| `phase0_phase1_phase2` | phase0, phase1, phase2 |
| `phase0_phase1_last` | phase0, phase1, last phase |
| `phase0_phase2_last` | phase0, phase2, last phase |
| `subtractions_phase1_phase2_last` | phase1-phase0, phase2-phase0, last-phase0 |

## 0. Access Check

Pillar-0 is gated on Hugging Face. Accept the terms for
`YalaLab/Pillar0-BreastMRI`, then verify that the same terminal account is
logged in.

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/hf auth login
```

```bash
HF_HOME=Benchmarking/outputs/model_cache ./.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
```

## 1. Smoke Test

Run this first. It loads the gated model, preprocesses one 3-channel 3D DCE
volume into the Pillar-0 shape, extracts one embedding, and writes a tiny output.

```bash
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

./.venv/bin/python -u Benchmarking/extract_pillar0_embeddings.py \
  --input-role phase0_phase1_last \
  --crop-mode whole \
  --max-patients 1 \
  --device auto \
  --torch-dtype float32 \
  --log-every 1 \
  > Benchmarking/outputs/logs/pillar0_whole_phase0_phase1_last_smoke.log 2>&1

status=$?
tail -80 Benchmarking/outputs/logs/pillar0_whole_phase0_phase1_last_smoke.log
printf "\nPillar-0 smoke exit status: %s\n" "$status"
```

The full `phase0_phase1_last` whole-volume run below will overwrite the
one-patient smoke output.

## 2. Whole-Volume Image-Only Extraction And Probes

Run whole-volume Pillar-0 first. This gives the first 3D benchmark result before
spending time on ROI crops.

```bash
set -euo pipefail
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

crop=whole
usable_flag=usable_for_multiphase_or_subtraction_benchmark

for role in phase0_phase1_phase2 phase0_phase1_last phase0_phase2_last subtractions_phase1_phase2_last; do
  ./.venv/bin/python -u Benchmarking/extract_pillar0_embeddings.py \
    --input-role "$role" \
    --crop-mode "$crop" \
    --device auto \
    --torch-dtype float32 \
    --log-every 25 \
    --checkpoint-every 25 \
    --resume \
    > "Benchmarking/outputs/logs/pillar0_${crop}_${role}_extract.log" 2>&1

  ./.venv/bin/python Benchmarking/train_embedding_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/pillar0_breastmri/${crop}_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag "$usable_flag" \
    --output-dir "Benchmarking/outputs/probes/pillar0_breastmri_${crop}_${role}_logreg"
done
```

## 3. Expert-ROI Image-Only Extraction And Probes

Run this after the whole-volume results have been checked.

```bash
set -euo pipefail
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

crop=expert_roi
usable_flag=usable_for_expert_roi_benchmark

for role in phase0_phase1_phase2 phase0_phase1_last phase0_phase2_last subtractions_phase1_phase2_last; do
  ./.venv/bin/python -u Benchmarking/extract_pillar0_embeddings.py \
    --input-role "$role" \
    --crop-mode "$crop" \
    --device auto \
    --torch-dtype float32 \
    --log-every 25 \
    --checkpoint-every 25 \
    --resume \
    > "Benchmarking/outputs/logs/pillar0_${crop}_${role}_extract.log" 2>&1

  ./.venv/bin/python Benchmarking/train_embedding_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/pillar0_breastmri/${crop}_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag "$usable_flag" \
    --output-dir "Benchmarking/outputs/probes/pillar0_breastmri_${crop}_${role}_logreg"
done
```

## 4. Image Plus Clinical

These four image-plus-clinical runs are the Pillar-0 multimodal headline set:
raw 3-channel DCE, subtraction triplet, and the matching expert-ROI versions.

```bash
set -euo pipefail

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/pillar0_breastmri/whole_phase0_phase1_last/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_multiphase_or_subtraction_benchmark \
  --output-dir Benchmarking/outputs/probes/pillar0_breastmri_whole_phase0_phase1_last_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/pillar0_breastmri/whole_subtractions_phase1_phase2_last/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_multiphase_or_subtraction_benchmark \
  --output-dir Benchmarking/outputs/probes/pillar0_breastmri_whole_subtractions_phase1_phase2_last_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/pillar0_breastmri/expert_roi_phase0_phase1_last/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_phase0_phase1_last_image_clinical_logreg

./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
  --embeddings Benchmarking/outputs/embeddings/pillar0_breastmri/expert_roi_subtractions_phase1_phase2_last/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_expert_roi_benchmark \
  --output-dir Benchmarking/outputs/probes/pillar0_breastmri_expert_roi_subtractions_phase1_phase2_last_image_clinical_logreg
```

## 5. Summary

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py Benchmarking/outputs/probes/*
./.venv/bin/python Benchmarking/build_cross_model_comparison.py
```
