# RadFM Feasibility Check

RadFM is the next unresolved model in the registry, but it should not be
downloaded yet. The official release is a full 13B multimodal generative model,
not a small embedding checkpoint. Its documentation asks for an NVIDIA A100
with 80 GB of memory for inference, and its released code uses
`transformers==4.28.1` rather than the current project environment.

The Hugging Face repository offers checkpoint packaging of roughly 50 GB. A
download plus extracted model can temporarily require about 100 GB, although
the exact amount depends on which archive route is used. The project disk has
enough room, but accelerator memory and software compatibility must be checked
first.

Sources:

- <https://github.com/chaoyi-wu/RadFM>
- <https://huggingface.co/chaoyi-wu/RadFM>

## 1. Run The Resource Audit

Run this from the project root. It is lightweight and does not download model
weights:

```bash
mkdir -p Benchmarking/outputs/logs

./.venv/bin/python - <<'PY' | tee Benchmarking/outputs/logs/radfm_resource_audit.txt
import shutil

import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("torch.version.hip:", torch.version.hip)
print("accelerator available through torch.cuda API:", torch.cuda.is_available())
print("accelerator count:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"accelerator {index}: {properties.name}")
    print(f"  total memory: {properties.total_memory / 2**30:.1f} GiB")
    try:
        free, total = torch.cuda.mem_get_info(index)
        print(f"  currently free: {free / 2**30:.1f} GiB")
        print(f"  currently visible total: {total / 2**30:.1f} GiB")
    except Exception as error:
        print(f"  free-memory query unavailable: {error}")

disk = shutil.disk_usage("Benchmarking")
print(f"Benchmarking disk free: {disk.free / 2**30:.1f} GiB")
print(f"Benchmarking disk total: {disk.total / 2**30:.1f} GiB")
PY
```

## 2. Decision Rule

- If an NVIDIA accelerator with approximately 80 GB of VRAM is available, set
  up RadFM in a separate legacy environment and validate the official full
  inference path before writing an extractor.
- If substantially less memory is available, do not download the checkpoint
  yet. The official full-model benchmark is not feasible on that accelerator.
- A custom vision-only loader could be investigated separately, but it would
  not be the official full RadFM inference path. It would still require the
  large checkpoint and must be labelled and validated as a model-specific
  adapter before it can enter the benchmark.

Do not install `transformers==4.28.1` into the main `.venv`; that would break
the environment used by the completed foundation-model extractors.

## 3. Audit Result

Completed on 2026-09-21:

```text
torch: 2.11.0+rocm7.2
transformers: 5.16.1
accelerator 0: AMD Radeon PRO W7900
total accelerator memory: 45.0 GiB
system memory: 30 GiB
Benchmarking disk free: 459.7 GiB
```

The official full-model route is **resource blocked** on this machine. Disk
capacity is sufficient, but the accelerator vendor and memory, host memory,
and software stack do not match the supported release. No checkpoint was
downloaded.

A vision-only implementation would require custom selective loading from the
full checkpoint. Because the authors do not release or document that path, it
would be a separate model-specific engineering experiment rather than an
official RadFM benchmark. It is not part of the primary benchmark matrix.
