# Final experiment package

Recorded results, figures, and provenance for the multimodal UAV detection project. The experiments combine vision, audio, and radar and evaluate fusion under degraded and missing-sensor conditions.

## Contents

- `figures/` — performance, robustness, gating, latency, and example plots
- `tables/` and `metrics/` — numeric results
- `demo/` — a CPU replay demo with packaged assets
- `provenance/`, `metadata/`, and `verification/` — run records and checks
- `logs/` — experiment and packaging logs

[`README_CHECKLIST.txt`](README_CHECKLIST.txt) records the packaged run and selected configuration. Raw and processed datasets are not included in this folder.

## Rebuild

From the repository root, with the datasets and `uavmm` environment ready:

```powershell
conda activate uavmm
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
python tools/make_figures.py
python tools/verify_isef_package.py
```

The recorded run used Windows, Python 3.11.14, PyTorch 2.10.0+cu128, CUDA 12.8, and an NVIDIA GeForce RTX 5070 Ti. Consult the metadata files for the full environment and data state.
