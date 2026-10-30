# UAV-MM

A research project on detecting drones by combining vision, audio, and radar. It compares individual sensor models with fixed and condition-adaptive fusion, including degraded and missing-sensor cases.

[View the demo](https://uav-detect-rsef-isr-project.vercel.app)

The training pipeline uses DroneDetectionThesis for vision and audio, and the TSMS drone dataset for radar. Results, figures, and run metadata are included in [`final_isef_package/`](final_isef_package/). The browser demo replays precomputed outputs; it does not run live inference.

## Try the local demo

The Streamlit demo runs without model weights in Poster Mode:

```sh
conda env create -f environment_demo.yml
conda activate uav-demo
```

On macOS, run `bash scripts/run_demo.sh`. On Windows, run `powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1`.

[Demo instructions](docs/README_DEMO.md) cover optional local inference, controls, and verification. The separate [Next.js demo](vercel_poster_demo/README.md) is used for the web presentation.

## Reproduce the experiments

The training workflow targets Windows with Conda and CUDA. Download the datasets separately into `data/raw/dronedetect_thesis` and `data/raw/tsms_drone`.

```powershell
conda env create -f environment.yml
conda activate uavmm
powershell -ExecutionPolicy Bypass -File .\run_all.ps1
```

The pipeline checks data and split integrity, trains the models, and writes evaluation artifacts to `results/`. It stops if required data or checks are missing. See [pipeline details](docs/PIPELINE.md) for individual commands, expected outputs, and reproducibility notes.

## Repository guide

- `src/` — preprocessing, datasets, models, training, and evaluation
- `experiments/` — experiment settings
- `demo_app/` — local Streamlit demo
- `vercel_poster_demo/` — web demo
- `final_isef_package/` — packaged results, figures, and provenance
- `presentation_assets/` — presentation exports and their source manifest

Use the recorded metrics and run metadata when describing results. Replay visuals and synthetic degradation scenarios should not be presented as new live measurements.
