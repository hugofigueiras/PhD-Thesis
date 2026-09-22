# Jolia Cross-Modality Benchmark Commands

Jolia is a 3D **CT** foundation model trained on adult chest and abdominal CT.
Its model card explicitly says it is not expected to generalize to other
modalities. This is therefore an advisor-requested MRI transfer stress test,
not a modality-matched entry in the primary breast-MRI leaderboard.

The pCR task, official MAMA-MIA train/test split, logistic probe, validation
selection, and clinical variables remain unchanged. The only model-specific
piece is the input adapter:

1. robustly scale each MRI volume using its 0.5th and 99.5th percentiles;
2. fit the full volume into a `192 x 192 x 192` cube without changing aspect ratio;
3. center-pad the remaining dimensions;
4. repeat that one MRI channel across Jolia's 11 fixed CT-window channels;
5. extract only the 576-D global embedding, not CT organ-query embeddings.

Outputs are tagged `cross_modality_stress_test` and saved under
`Benchmarking/outputs/embeddings/jolia_cross_modality/`.

## 0. Accept Access Terms

Open the model page and request/accept access using the same Hugging Face
account already configured in `Benchmarking/outputs/model_cache`:

<https://huggingface.co/raidium/Jolia>

Then verify both the account and repository access:

```bash
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub

./.venv/bin/python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"

./.venv/bin/python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='raidium/Jolia', filename='config.json'))"
```

The second command must print a local `config.json` path. A `403` means the
Jolia-specific terms have not been approved for the account yet.

## 1. One-Patient CUDA Smoke Test

```bash
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

./.venv/bin/python -u Benchmarking/extract_jolia_embeddings.py \
  --input-role phase1 \
  --crop-mode whole \
  --max-patients 1 \
  --device auto \
  --log-every 1 \
  --checkpoint-every 1 \
  > Benchmarking/outputs/logs/jolia_cross_modality_whole_phase1_smoke.log 2>&1

status=$?
tail -80 Benchmarking/outputs/logs/jolia_cross_modality_whole_phase1_smoke.log
printf '\nJolia smoke exit status: %s\n' "$status"
```

Continue only if the status is `0` and the summary reports `embedding_dim: 576`.
The full phase-1 extraction uses `--resume`, so the valid smoke embedding is
reused rather than discarded.

## 2. Predeclared Image-Only Transfer Screen

These are the same high-value inputs used in the standardized benchmarks:
early post-contrast phase 1, phase 2 minus pre-contrast, and the expert-ROI
version of that subtraction. They are selected in advance, not after seeing
Jolia's results.

```bash
set -euo pipefail
mkdir -p Benchmarking/outputs/logs
export HF_HOME=Benchmarking/outputs/model_cache
export HF_HUB_CACHE=Benchmarking/outputs/model_cache/hub
export TOKENIZERS_PARALLELISM=false

for item in \
  whole:phase1:usable_for_multiphase_or_subtraction_benchmark \
  whole:phase2_minus_phase0:usable_for_multiphase_or_subtraction_benchmark \
  expert_roi:phase2_minus_phase0:usable_for_expert_roi_benchmark
do
  IFS=: read -r crop role usable_flag <<< "$item"

  ./.venv/bin/python -u Benchmarking/extract_jolia_embeddings.py \
    --input-role "$role" \
    --crop-mode "$crop" \
    --usable-flag "$usable_flag" \
    --device auto \
    --log-every 25 \
    --checkpoint-every 25 \
    --resume \
    > "Benchmarking/outputs/logs/jolia_cross_modality_${crop}_${role}_extract.log" 2>&1

  ./.venv/bin/python Benchmarking/train_embedding_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/jolia_cross_modality/${crop}_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag "$usable_flag" \
    --output-dir "Benchmarking/outputs/probes/jolia_cross_modality_${crop}_${role}_logreg"
done
```

Monitor the currently running extraction from another terminal with:

```bash
tail -f Benchmarking/outputs/logs/jolia_cross_modality_whole_phase1_extract.log
```

## 3. Selected MRI Feature Fusion

This tests whether static early enhancement and a later enhancement change are
complementary. It concatenates saved patient embeddings; Jolia remains frozen.

```bash
set -euo pipefail

./.venv/bin/python Benchmarking/fuse_embedding_tables.py \
  --input phase1=Benchmarking/outputs/embeddings/jolia_cross_modality/whole_phase1/patient_embeddings.csv \
  --input phase2sub=Benchmarking/outputs/embeddings/jolia_cross_modality/whole_phase2_minus_phase0/patient_embeddings.csv \
  --output Benchmarking/outputs/embeddings/jolia_cross_modality/whole_selected_fusion/patient_embeddings.csv

./.venv/bin/python Benchmarking/train_embedding_probe.py \
  --embeddings Benchmarking/outputs/embeddings/jolia_cross_modality/whole_selected_fusion/patient_embeddings.csv \
  --feature-prefix emb_ \
  --usable-flag usable_for_multiphase_or_subtraction_benchmark \
  --output-dir Benchmarking/outputs/probes/jolia_cross_modality_whole_selected_fusion_logreg
```

## 4. Image Plus Clinical Probes

```bash
set -euo pipefail

for item in \
  whole:phase1:usable_for_multiphase_or_subtraction_benchmark \
  whole:phase2_minus_phase0:usable_for_multiphase_or_subtraction_benchmark \
  whole:selected_fusion:usable_for_multiphase_or_subtraction_benchmark \
  expert_roi:phase2_minus_phase0:usable_for_expert_roi_benchmark
do
  IFS=: read -r crop role usable_flag <<< "$item"

  ./.venv/bin/python Benchmarking/train_image_clinical_probe.py \
    --embeddings "Benchmarking/outputs/embeddings/jolia_cross_modality/${crop}_${role}/patient_embeddings.csv" \
    --feature-prefix emb_ \
    --usable-flag "$usable_flag" \
    --output-dir "Benchmarking/outputs/probes/jolia_cross_modality_${crop}_${role}_image_clinical_logreg"
done
```

## 5. Separate Stress-Test Summary

Keep these results separate from the modality-matched leaderboard:

```bash
./.venv/bin/python Benchmarking/summarize_probe_runs.py \
  Benchmarking/outputs/probes/jolia_cross_modality_* \
  --strict \
  --output-csv Benchmarking/outputs/summaries/jolia_cross_modality_probe_summary.csv
```

Do not expand to all seven phase inputs until this predeclared screen has been
checked. A clearly negative screen is already a valid generalization result;
running more MRI variants would not make the CT model modality-matched.
