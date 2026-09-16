# MedSigLIP First-Pass Benchmark Summary

MedSigLIP first-pass benchmarking is complete on the official MAMA-MIA
train/test split. This includes 13 image-only runs and 4 image-plus-clinical
runs using the same protocol as RadioDINO, BiomedCLIP, and Curia.

## Best Image-Only Results

| Scope | Input | AUROC | AP | Balanced accuracy |
|---|---|---:|---:|---:|
| Best whole-volume AUROC | last phase | 0.593 | 0.378 | 0.546 |
| Best whole-volume AP | phase2 - phase0 | 0.585 | 0.390 | 0.544 |
| Best whole-volume balanced accuracy | phase1 - phase0 | 0.587 | 0.363 | 0.562 |
| Best ROI AUROC | expert-ROI phase2 - phase0 | 0.621 | 0.403 | 0.565 |
| Best ROI AP/balanced accuracy | selected ROI fusion | 0.619 | 0.404 | 0.578 |

## Image-Plus-Clinical Results

| Input | AUROC | AP | Balanced accuracy |
|---|---:|---:|---:|
| whole phase1 + clinical | 0.706 | 0.483 | 0.579 |
| whole phase2 - phase0 + clinical | 0.699 | 0.482 | 0.632 |
| whole all-DCE fusion + clinical | 0.639 | 0.426 | 0.621 |
| expert-ROI phase2 - phase0 + clinical | 0.703 | 0.481 | 0.653 |

## Interpretation

MedSigLIP has meaningful image-only pCR signal, especially after expert ROI
cropping. ROI improves the best image-only AUROC from 0.593 to 0.621 and gives
the best image-only balanced accuracy at 0.578.

Adding clinical variables helps substantially. The best MedSigLIP multimodal
AUROC and AP come from whole phase1 plus clinical: AUROC 0.706 and AP 0.483.
The best MedSigLIP balanced accuracy comes from expert-ROI phase2-minus-phase0
plus clinical: 0.653.

Compared with the current global benchmarks, MedSigLIP is competitive but not
the leading model. Curia remains the strongest image-only model, and BiomedCLIP
whole phase1 plus clinical remains the strongest multimodal AUROC/AP result.
MedSigLIP expert-ROI phase2-minus-phase0 plus clinical is close to Curia on
balanced accuracy, but still slightly lower.

## Saved Tables

```text
Benchmarking/outputs/summaries/medsiglip_first_pass_probe_summary.csv
Benchmarking/outputs/summaries/cross_model_first_pass_comparison.csv
Benchmarking/outputs/summaries/cross_model_all_results.md
Benchmarking/outputs/summaries/cross_model_best_by_metric.csv
```
