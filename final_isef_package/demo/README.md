# Offline demo

A CPU-only replay of packaged UAV detection outputs. It uses the small files in `assets/`, not live model inference.

From the repository root, create and activate a Python virtual environment, then run:

```sh
pip install -r final_isef_package/demo/requirements_cpu.txt
streamlit run final_isef_package/demo/app.py
```

The screenshot helper runs separately:

```sh
python final_isef_package/demo/run_demo.py
```

It writes `final_isef_package/demo/output/demo_screenshot.png`. The demo is intended for Windows and macOS, including Apple Silicon.
