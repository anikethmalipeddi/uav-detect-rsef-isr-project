$CondaExe = Join-Path $env:USERPROFILE "miniforge3\\Scripts\\conda.exe"
& $CondaExe run -n uavmm python -m src.eval.eval_unimodal --episodes data/processed/episodes/episodes.csv
& $CondaExe run -n uavmm python -m src.eval.eval_fusion --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
& $CondaExe run -n uavmm python -m src.eval.ablations --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
& $CondaExe run -n uavmm python -m src.eval.missing_modality --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
& $CondaExe run -n uavmm python -m src.eval.stress_tests --episodes data/processed/episodes/episodes.csv --features data/processed/episodes/features_cache.npz
