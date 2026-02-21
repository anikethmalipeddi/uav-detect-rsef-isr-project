param(
  [string]$VisionIn = "data/raw/dronedetect_thesis/Data/Video_V",
  [string]$VisionInIR = "data/raw/dronedetect_thesis/Data/Video_IR",
  [string]$AudioIn = "data/raw/dronedetect_thesis/Data/Audio",
  [string]$RadarIn = "data/raw/tsms_drone"
)

$CondaExe = Join-Path $env:USERPROFILE "miniforge3\\Scripts\\conda.exe"
& $CondaExe run -n uavmm python -m src.preprocess.extract_frames --input $VisionIn --output data/processed/vision_frames --fps 10
& $CondaExe run -n uavmm python -m src.preprocess.extract_frames --input $VisionInIR --output data/processed/vision_frames --fps 10
& $CondaExe run -n uavmm python -m src.preprocess.audio_windows --input $AudioIn --output data/processed/audio_windows
& $CondaExe run -n uavmm python -m src.preprocess.radar_extract_tsms --input $RadarIn --output data/processed/radar_rd
& $CondaExe run -n uavmm python -m src.datasets.episode_builder --vision-manifest data/processed/vision_frames/manifest.csv --audio-manifest data/processed/audio_windows/manifest.csv --radar-manifest data/processed/radar_rd/manifest.csv --out data/processed/episodes/episodes.csv --target-count 2000
& $CondaExe run -n uavmm python tools/dataset_stats.py --root data/raw --out results/dataset_stats.json
