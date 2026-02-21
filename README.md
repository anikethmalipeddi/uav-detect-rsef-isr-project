# UAV-MM: Condition-Adaptive Multimodal UAV Detection

Offline, reproducible week-1 implementation for multimodal UAV presence detection with vision, audio, and radar signals.

## Scope
- Windows-native workflow.
- Core datasets only in week 1:
  - DroneDetectionThesis (vision + audio)
  - TSMS Drone dataset (radar)
- Includes a lightweight cross-platform ISEF demo app:
  - `Poster Mode` (offline replay, no weights)
  - `Full Mode` (optional local weights in `models/`, auto-fallback)

## Project Layout
```text
data/
  raw/
  processed/
results/
experiments/
src/
  preprocess/
  datasets/
  models/
  train/
  eval/
  utils/
```

## Environment Setup (Windows)
1. Install Miniforge.
2. Create environment:
   ```powershell
   conda env create -f environment.yml
   conda activate uavmm
   ```
3. Verify CUDA:
   ```powershell
   nvidia-smi
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

Note: `radarsimpy` is not pinned in `environment.yml` because Windows wheels are not consistently available. The default pipeline in `run_all.ps1` uses only real TSMS radar data and fails on missing required data.

## One-Command Run (Fail-Fast)
Run the full end-to-end pipeline from repo root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

What this script enforces:
- strict TSMS verification (`results/tsms_state.txt`, `results/tsms_state.json`)
- full DroneDetectionThesis label conversion + overlays + label stats
- split integrity check (`results/split_integrity_report.txt`)
- unimodal + fusion training and evaluation
- explicit fusion-advantage check (`results/fusion_advantage_check.txt`) that fails if `F-GATE` does not beat `F-AVG` and `F-STATIC` on test PR-AUC
- required tables, stress tests, figures, significance, and run metadata
- immediate failure on missing artifacts or invalid intermediate outputs

Fast validation only:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_sanity.ps1
```

## Live Demo (Windows + macOS M2)
- Demo app source: `demo_app/app.py`
- Demo instructions: `docs/README_DEMO.md`
- Windows:
  ```powershell
  conda env create -f environment_demo.yml
  conda activate uav-demo
  powershell -ExecutionPolicy Bypass -File .\scripts\run_demo.ps1
  ```
- macOS:
  ```bash
  conda env create -f environment_demo.yml
  conda activate uav-demo
  bash scripts/run_demo.sh
  ```

## Data Setup
1. Clone DroneDetectionThesis into `data/raw/dronedetect_thesis`.
2. Download TSMS data into `data/raw/tsms_drone`.

The scripts expect raw data under these exact folders.

For TSMS full download with resume support:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_tsms_full.ps1 -Root data/raw/tsms_drone
```

## Day 1-2 Pipeline Commands
```powershell
python -m src.preprocess.inspect_mat --mat data/raw/dronedetect_thesis/Data/Video_V/V_DRONE_001_LABELS.mat
python -m src.preprocess.extract_frames --input data/raw/dronedetect_thesis/Data/Video_V --output data/processed/vision_frames --fps 10
python -m src.preprocess.extract_frames --input data/raw/dronedetect_thesis/Data/Video_IR --output data/processed/vision_frames --fps 10
python -m src.preprocess.audio_windows --input data/raw/dronedetect_thesis/Data/Audio --output data/processed/audio_windows
python -m src.preprocess.radar_extract_tsms --input data/raw/tsms_drone --output data/processed/radar_rd
python -m src.datasets.episode_builder --vision-manifest data/processed/vision_frames/manifest.csv --audio-manifest data/processed/audio_windows/manifest.csv --radar-manifest data/processed/radar_rd/manifest.csv --out data/processed/episodes/episodes.csv --target-count 2000
```

## Training Commands
```powershell
python -m src.train.train_vision_yolo --data data/processed/vision_yolo/dataset.yaml --model yolov8n.pt --epochs 200
python -m src.train.train_audio --episodes data/processed/episodes/episodes.csv
python -m src.train.train_radar --episodes data/processed/episodes/episodes.csv
python -m src.train.export_vision_presence --episodes data/processed/episodes/episodes.csv
python -m src.train.train_fusion --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
```

## Evaluation Commands
```powershell
python -m src.eval.eval_unimodal --episodes data/processed/episodes/episodes.csv
python -m src.eval.ablations --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
python -m src.eval.missing_modality --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
python -m src.eval.stress_tests --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
```

## Required Output Artifacts
- `results/metrics_unimodal.csv`
- `results/metrics_fusion.csv`
- `results/ablation_table.csv`
- `results/missing_modality.csv`
- `results/stress_audio_snr.csv`
- `results/stress_vision_corruptions.csv`
- `results/stress_radar_clutter.csv`
- `results/latency_summary.json`
- `results/predictions_fusion.csv`
- `results/significance_pr_auc.txt`
- `results/fusion_advantage_check.txt`
- `results/split_integrity_report.txt`
- `results/run_metadata.json`

## Reproducibility
- Set fixed seeds with `src.utils.seed.set_seed`.
- Keep train/val/test split fixed in `episodes.csv`.
- Record commit hash for final runs:
  ```powershell
  git rev-parse HEAD
  ```
- Run metadata is written to `results/run_metadata.json` with machine, torch/cuda, git hash, dataset state, and warnings.


