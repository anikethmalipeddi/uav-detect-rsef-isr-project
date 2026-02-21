# ISEF Live Demo (Cross-Platform)

This demo has two runtime modes:

- `Poster Mode` (default): fully offline replay using committed `demo_assets/` + `demo_outputs/`; no weights needed.
- `Full Mode` (optional): attempts local live inference only when local weights exist in `models/` (gitignored). If unavailable, it auto-falls back to Poster Mode.

The app always shows a status badge indicating `Poster Mode: Precomputed outputs` or `Full Mode: Live inference`.

## Quick Start

### Windows (Conda)

```powershell
conda env create -f environment_demo.yml
conda activate uav-demo
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

Optional Full Mode:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1 -Mode full
```

### macOS Apple Silicon (M2) with Conda

```bash
conda env create -f environment_demo.yml
conda activate uav-demo
bash scripts/run_demo.sh
```

Optional Full Mode:

```bash
bash scripts/run_demo.sh full
```

### macOS Apple Silicon (M2) with pip

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_demo.txt
streamlit run demo_app/app.py
```

## Demo Controls and Tabs

- `Live Inference`: video overlays, audio waveform/spectrogram, radar heatmap, per-modality and fused scores, gating over time.
- `One-click Judge Demo`: scripted sequence `Clean -> Audio Degraded -> Vision Degraded -> Radar Clutter -> Missing Audio`.
- `Robustness Lab`: SNR/vision/radar sliders + missing modality toggles with live replay updates.
- `Compare Models`: unimodal and fusion metrics from final run tables.
- `Latency / Deploy`: latency summary from `final_isef_package/tables/latency_summary.json`.
- `Export`: creates poster-ready files in `final_isef_package/demo_exports/<timestamp>/`.

## Judge Mode + Screenshot

- Use the `Judge Mode` toggle in the app header for presentation layout.
- Use `Take Screenshot for Poster` (Live tab) or `Export poster bundle` (Export tab).
- Exports include:
  - `overlay_frame.png`
  - `gating_weights_chart.png`
  - `confidence_panel.png`
  - `gating_weights.csv`
  - `predictions_per_frame.csv`
  - `run_settings.json`
  - `caption.txt`

## Verification

Run demo checks:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/verify_demo.ps1
```

```bash
bash scripts/verify_demo.sh
```

## Notes on Full Mode

- Place local weights in `models/` with these filenames:
  - `vision_model.pt`
  - `audio_model.pt`
  - `radar_model.pt`
  - `fusion_model.pt`
- If weights or runtime dependencies are missing, the app auto-falls back to Poster Mode without crashing.

