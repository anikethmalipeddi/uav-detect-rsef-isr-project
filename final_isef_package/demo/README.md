# ISEF Demo (CPU Offline)

This demo is fully offline and runs on macOS Apple Silicon (M2) CPU or Windows CPU.
It uses tiny packaged assets in `assets/` and produces a poster screenshot in `output/demo_screenshot.png`.

## macOS M2 (CPU)
```bash
python3 -m venv .venv_demo
source .venv_demo/bin/activate
pip install -r final_isef_package/demo/requirements_cpu.txt
streamlit run final_isef_package/demo/app.py
python final_isef_package/demo/run_demo.py
```

## Windows (CPU)
```powershell
python -m venv .venv_demo
.\.venv_demo\Scripts\Activate.ps1
pip install -r final_isef_package/demo/requirements_cpu.txt
streamlit run final_isef_package/demo/app.py
python final_isef_package/demo/run_demo.py
```
