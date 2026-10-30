# Local UAV demo

The Streamlit app opens in Poster Mode, replaying committed assets and outputs without model weights. Optional Full Mode attempts local inference when the required weights are available and falls back to replay when they are not. Check the mode badge before describing what the demo is showing.

## Run

From the repository root:

```sh
conda env create -f environment_demo.yml
conda activate uav-demo
```

On Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_demo.ps1
```

On macOS:

```sh
bash scripts/run_demo.sh
```

Without Conda, create a Python virtual environment, install `requirements_demo.txt`, and run `streamlit run demo_app/app.py`.

## Local inference

Place the local weights in the ignored `models/` directory as `vision_model.pt`, `audio_model.pt`, `radar_model.pt`, and `fusion_model.pt`.

Use `scripts/run_demo.sh full` on macOS, or add `-Mode full` to the Windows launcher. Missing weights or dependencies cause the app to use Poster Mode.

## Controls and exports

The app includes video overlays, audio and radar views, modality scores, and gating weights. The scripted judge sequence moves through clean input, degraded audio, degraded vision, radar clutter, and missing audio.

The Robustness Lab changes replay conditions; Compare Models reads the recorded metrics. Judge Mode adjusts the presentation layout. The Export tab saves overlays, gating charts, score panels, CSVs, settings, and captions under `final_isef_package/demo_exports/`.

## Verify

Run `bash scripts/verify_demo.sh` on macOS, or `powershell -ExecutionPolicy Bypass -File scripts/verify_demo.ps1` on Windows.
