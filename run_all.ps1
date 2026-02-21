param(
  [string]$CondaEnv = "uavmm",
  [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
  [int]$Seed = 42,
  [int]$TargetEpisodes = 4000,
  [int]$AudioEpochs = 50,
  [int]$RadarEpochs = 50,
  [int]$GateEpochs = 260,
  [double]$GateLr = 0.0005,
  [double]$GateDropOne = 0.10,
  [double]$GateDropTwo = 0.03,
  [double]$GateAuxWeight = 0.0,
  [int]$FusionMaxConfigs = 16,
  [int]$YoloEpochs = 100,
  [int]$YoloImgSz = 640,
  [int]$YoloFrameStep = 4,
  [string]$YoloRunName = "uavmm_yolov8n"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
# Avoid treating native stderr text (e.g., tqdm progress) as terminating errors.
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}

if (-not (Test-Path $CondaExe)) {
  throw "conda executable not found: $CondaExe"
}

New-Item -ItemType Directory -Path "results/logs" -Force | Out-Null
if (Test-Path "results/fallback_warnings.txt") {
  Remove-Item -Force "results/fallback_warnings.txt"
}

$configSnapshot = [ordered]@{
  conda_env = $CondaEnv
  conda_exe = $CondaExe
  seed = $Seed
  target_episodes = $TargetEpisodes
  audio_epochs = $AudioEpochs
  radar_epochs = $RadarEpochs
  gate_epochs = $GateEpochs
  gate_lr = $GateLr
  gate_drop_one = $GateDropOne
  gate_drop_two = $GateDropTwo
  gate_aux_weight = $GateAuxWeight
  fusion_max_configs = $FusionMaxConfigs
  yolo_epochs = $YoloEpochs
  yolo_imgsz = $YoloImgSz
  yolo_frame_step = $YoloFrameStep
  yolo_run_name = $YoloRunName
}
$configSnapshot | ConvertTo-Json -Depth 4 | Set-Content -Path "results/run_config_snapshot.json" -Encoding UTF8

# Clean key outputs to ensure this run regenerates final artifacts.
$staleArtifacts = @(
  "results/figures",
  "results/label_viz",
  "results/metrics_unimodal.csv",
  "results/metrics_fusion.csv",
  "results/ablation_table.csv",
  "results/missing_modality.csv",
  "results/stress_audio_snr.csv",
  "results/stress_vision_corruptions.csv",
  "results/stress_radar_clutter.csv",
  "results/latency_summary.json",
  "results/preds_fusion.csv",
  "results/predictions_fusion.csv",
  "results/significance_pr_auc.txt",
  "results/fusion_tuning.csv",
  "results/fusion_best_config.json",
  "results/split_integrity_report.txt",
  "results/split_integrity_report.json",
  "results/label_stats.json",
  "results/label_stats.txt",
  "results/label_per_clip_summary.csv",
  "results/dronedetect_state.txt",
  "results/dronedetect_state.json",
  "results/tsms_state.txt",
  "results/tsms_state.json",
  "results/tsms_download_complete.flag",
  "results/run_metadata.json",
  "results/config_used.yaml"
)
foreach ($artifact in $staleArtifacts) {
  if (Test-Path $artifact) {
    Remove-Item -Force -Recurse $artifact -ErrorAction SilentlyContinue
  }
}

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][scriptblock]$Body
  )
  $log = Join-Path "results/logs" ($Name + ".log")
  Write-Host "==> $Name"
  $exitCode = 0
  $prevErrorAction = $ErrorActionPreference
  try {
    # Native tools frequently write progress to stderr; use explicit exit-code checks instead.
    $ErrorActionPreference = "Continue"
    & $Body 2>&1 | Tee-Object -FilePath $log
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $prevErrorAction
  }
  if ($exitCode -ne 0) {
    throw "Step failed: $Name (exit code $exitCode). See $log"
  }
}

Invoke-Step "download_tsms_full" {
  & "powershell.exe" -NoProfile -ExecutionPolicy Bypass -File ".\scripts\download_tsms_full.ps1" `
    -Root "data/raw/tsms_drone" `
    -Extract `
    -DeleteArchive `
    -RetryCount 3 `
    -BackoffBaseSec 15 `
    -MinExtractedFiles 1000 `
    -CompleteFlag "results/tsms_download_complete.flag"
}

Invoke-Step "verify_tsms" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.verify_tsms `
    --root data/raw/tsms_drone `
    --out-txt results/tsms_state.txt `
    --out-json results/tsms_state.json `
    --complete-flag results/tsms_download_complete.flag `
    --min-total-gb 150.0 `
    --min-extracted-files-per-archive 1000
}

Invoke-Step "verify_dronedetect" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.verify_dronedetect `
    --root data/raw/dronedetect_thesis/Data `
    --min-video-v 285 `
    --min-video-ir 365 `
    --min-label-v 285 `
    --min-label-ir 365 `
    --min-audio 90 `
    --min-total-gb 0.30 `
    --expect-video-v 285 `
    --expect-video-ir 365 `
    --expect-label-v 285 `
    --expect-label-ir 365 `
    --expect-audio 90 `
    --out-txt results/dronedetect_state.txt `
    --out-json results/dronedetect_state.json
}

Invoke-Step "convert_mat_labels" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.convert_mat_labels `
    --mat-root data/raw/dronedetect_thesis/Data `
    --out-label-csv data/processed/vision_labels/labels_long.csv `
    --out-label-json data/processed/vision_labels/labels_long.jsonl `
    --out-video-csv data/processed/vision_labels/videos.csv `
    --out-stats results/label_stats.json `
    --out-stats-txt results/label_stats.txt `
    --overlay-dir results/label_viz `
    --overlay-samples 80 `
    --yolo-out data/processed/vision_yolo `
    --yolo-frame-step $YoloFrameStep `
    --seed $Seed
}

Invoke-Step "extract_frames" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.extract_frames `
    --input data/raw/dronedetect_thesis/Data `
    --output data/processed/vision_frames `
    --fps 10 `
    --manifest data/processed/vision_frames/manifest.csv
}

Invoke-Step "audio_windows" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.audio_windows `
    --input data/raw/dronedetect_thesis/Data/Audio `
    --output data/processed/audio_windows `
    --manifest data/processed/audio_windows/manifest.csv
}

Invoke-Step "radar_extract_tsms" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.radar_extract_tsms `
    --input data/raw/tsms_drone `
    --output data/processed/radar_rd_real `
    --manifest data/processed/radar_rd_real/manifest.csv
}

Invoke-Step "episode_builder" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.datasets.episode_builder `
    --vision-manifest data/processed/vision_frames/manifest.csv `
    --audio-manifest data/processed/audio_windows/manifest.csv `
    --radar-manifest data/processed/radar_rd_real/manifest.csv `
    --out data/processed/episodes/episodes.csv `
    --target-count $TargetEpisodes `
    --seed $Seed
}

Invoke-Step "split_integrity" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.split_integrity `
    --episodes data/processed/episodes/episodes.csv `
    --out-txt results/split_integrity_report.txt `
    --out-json results/split_integrity_report.json
}

Invoke-Step "train_vision_yolo" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.train.train_vision_yolo `
    --data data/processed/vision_yolo/dataset.yaml `
    --model yolov8n.pt `
    --epochs $YoloEpochs `
    --imgsz $YoloImgSz `
    --seed $Seed `
    --run-name $YoloRunName `
    --out-metrics results/metrics_vision.json
}

$VisionWeights = (
  & $CondaExe run --no-capture-output -n $CondaEnv python -c "import json; print(json.load(open('results/metrics_vision.json','r',encoding='utf-8'))['weights_path'])"
).Trim()
if (-not (Test-Path $VisionWeights)) {
  throw "Vision weights file not found after training: $VisionWeights"
}

Invoke-Step "export_vision_presence" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.train.export_vision_presence `
    --episodes data/processed/episodes/episodes.csv `
    --model $VisionWeights `
    --out data/processed/episodes/vision_presence.csv
}

Invoke-Step "train_audio" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.train.train_audio `
    --episodes data/processed/episodes/episodes.csv `
    --epochs $AudioEpochs `
    --seed $Seed `
    --out-ckpt results/checkpoints/best_audio.pt
}

Invoke-Step "train_radar" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.train.train_radar `
    --episodes data/processed/episodes/episodes.csv `
    --epochs $RadarEpochs `
    --seed $Seed `
    --out-ckpt results/checkpoints/best_radar.pt
}

Invoke-Step "tune_fusion" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.train.tune_fusion `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --vision-csv data/processed/episodes/vision_presence.csv `
    --audio-ckpt results/checkpoints/best_audio.pt `
    --radar-ckpt results/checkpoints/best_radar.pt `
    --seed $Seed `
    --epochs $GateEpochs `
    --lr-grid "$GateLr" `
    --drop-one-grid "$GateDropOne" `
    --drop-two-grid "$GateDropTwo" `
    --aux-weight-grid "$GateAuxWeight" `
    --max-configs $FusionMaxConfigs `
    --out-csv results/fusion_tuning.csv `
    --best-json results/fusion_best_config.json `
    --train-final
}

Invoke-Step "eval_unimodal" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.eval_unimodal `
    --episodes data/processed/episodes/episodes.csv `
    --audio-ckpt results/checkpoints/best_audio.pt `
    --radar-ckpt results/checkpoints/best_radar.pt `
    --vision-csv data/processed/episodes/vision_presence.csv `
    --out results/metrics_unimodal.csv
}

Invoke-Step "eval_fusion" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.eval_fusion `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --gate-ckpt results/checkpoints/fusion_gate.pt `
    --gate-noq-ckpt results/checkpoints/fusion_gate_noq.pt `
    --static-alpha results/checkpoints/fusion_static_alpha.npy `
    --out results/metrics_fusion.csv `
    --ablation-out results/ablation_table.csv `
    --pred-out results/preds_fusion.csv `
    --split test
}

Invoke-Step "check_fusion_advantage" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.check_fusion_advantage `
    --metrics results/metrics_fusion.csv `
    --out-txt results/fusion_advantage_check.txt `
    --metric pr_auc `
    --split test
}

Invoke-Step "eval_missing_modality" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.missing_modality `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --gate-ckpt results/checkpoints/fusion_gate.pt `
    --gate-noq-ckpt results/checkpoints/fusion_gate_noq.pt `
    --static-alpha results/checkpoints/fusion_static_alpha.npy `
    --out results/missing_modality.csv
}

Invoke-Step "eval_stress_tests" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.stress_tests `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --gate-ckpt results/checkpoints/fusion_gate.pt `
    --gate-noq-ckpt results/checkpoints/fusion_gate_noq.pt `
    --static-alpha results/checkpoints/fusion_static_alpha.npy `
    --seed $Seed
}

Invoke-Step "significance_pr_auc" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.significance_pr_auc `
    --pred results/preds_fusion.csv `
    --split test `
    --n-bootstrap 2000 `
    --seed $Seed `
    --out results/significance_pr_auc.txt
}

Invoke-Step "make_figures" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.make_figures `
    --fig-dir results/figures
}

Invoke-Step "sanity_checks" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.sanity_checks `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --out-json results/sanity_checks.json `
    --out-txt results/sanity_checks.txt
}

Invoke-Step "config_used" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.utils.write_config_used `
    --config-snapshot results/run_config_snapshot.json `
    --fusion-best results/fusion_best_config.json `
    --tsms-state results/tsms_state.json `
    --dronedetect-state results/dronedetect_state.json `
    --out results/config_used.yaml
}

Invoke-Step "run_metadata" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.utils.run_metadata `
    --out results/run_metadata.json `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --tsms-state results/tsms_state.json `
    --dronedetect-state results/dronedetect_state.json `
    --label-stats results/label_stats.json `
    --fusion-best results/fusion_best_config.json `
    --fusion-tuning results/fusion_tuning.csv `
    --fallback-warnings results/fallback_warnings.txt `
    --config-snapshot results/run_config_snapshot.json `
    --config-used results/config_used.yaml `
    --seed $Seed
}

Invoke-Step "assert_final_run_eligible" {
  if (Test-Path "results/fallback_warnings.txt") {
    $warn = Get-Content "results/fallback_warnings.txt" -ErrorAction SilentlyContinue
    throw "Fallback warnings detected; FINAL run is invalid. Details: $warn"
  }
  if (-not (Test-Path "results/run_metadata.json")) {
    throw "Missing results/run_metadata.json after metadata step."
  }
  $meta = Get-Content "results/run_metadata.json" -Raw | ConvertFrom-Json
  if (-not $meta.final_run_eligible) {
    throw "run_metadata indicates final_run_eligible=false. Check dataset verification and warnings."
  }
}

Invoke-Step "build_final_isef_package" {
  & "powershell.exe" -NoProfile -ExecutionPolicy Bypass -File ".\scripts\build_final_isef_package.ps1"
}

Write-Host "Pipeline complete. Artifacts are in results/"
