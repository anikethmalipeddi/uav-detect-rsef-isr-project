$CondaExe = Join-Path $env:USERPROFILE "miniforge3\\Scripts\\conda.exe"
& $CondaExe run -n uavmm python -m src.train.train_audio --episodes data/processed/episodes/episodes.csv --epochs 200
& $CondaExe run -n uavmm python -m src.train.train_radar --episodes data/processed/episodes/episodes.csv --epochs 200
& $CondaExe run -n uavmm python -m src.train.export_vision_presence --episodes data/processed/episodes/episodes.csv --model runs/detect/uavmm_yolov8n/weights/best.pt
& $CondaExe run -n uavmm python -m src.train.train_fusion --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
