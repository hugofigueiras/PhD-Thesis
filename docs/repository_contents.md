# Repository Contents And Exclusions

This repository is a public, lightweight research-workflow repository. It is not
the primary storage location for medical images or generated model artifacts.

## Included

- reconstruction scripts;
- QC summary JSON files;
- dataset release documentation;
- Hugging Face upload preparation script;
- foundation-model registry;
- benchmark command notes and training/extraction scripts;
- current lightweight benchmark result summaries;
- PCR prediction scripts and compact comparison tables.

## Excluded

- original DICOM files;
- reconstructed NIfTI image volumes;
- local copies of original datasets;
- Hugging Face staging hardlinks;
- generated foundation-model embeddings;
- trained model checkpoints;
- logistic-probe model binary files;
- nnUNet preprocessed folders and checkpoints;
- W&B run directories;
- Hugging Face model caches;
- credential/token/cache files;
- copyrighted paper PDFs.

## Data Location

The public reconstructed dataset is hosted on Hugging Face:

https://huggingface.co/datasets/hfigueiras/LMBM

The GitHub repository should cite and link to the Hugging Face dataset rather
than duplicate the image volumes.
