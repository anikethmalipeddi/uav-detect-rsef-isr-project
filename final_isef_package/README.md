# ISEF Final Package

This package contains the final, curated artifacts for the condition-adaptive multimodal UAV detection project. The system fuses **vision**, **audio**, and **radar** evidence with adaptive gating and modality masking to detect UAV presence robustly under degradation and missing-modality conditions.

## Reproduce Final Run
1. Activate environment:
   - `conda activate uavmm`
2. Run end-to-end pipeline:
   - `powershell -ExecutionPolicy Bypass -File .\run_all.ps1`
3. Rebuild poster figures:
   - `python tools/make_figures.py`
4. Verify package completeness:
   - `python tools/verify_isef_package.py`

## Artifact Usage Map
- `figures/`: poster and presentation plots (performance, robustness, gating, latency, qualitative).
- `tables/` and `metrics/`: numeric tables for poster text and claims.
- `demo/`: CPU-friendly offline demo for judges (macOS M2 + Windows).
- `provenance/`: reproducibility metadata (environment, data manifest, run metadata).
- `logs/`: final run and package verification logs.

## Hardware / Software Summary
- Machine: `DESKTOP-RLP7CRM`
- OS: `Windows-10-10.0.26200-SP0`
- Python: `3.11.14 | packaged by conda-forge | (main, Jan 26 2026, 23:39:55) [MSC v.1944 64 bit (AMD64)]`
- Torch: `2.10.0+cu128`
- CUDA: `12.8`
- GPU: `NVIDIA GeForce RTX 5070 Ti`
