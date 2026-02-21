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
Set-Location (Split-Path -Parent $PSScriptRoot)

& "powershell.exe" -NoProfile -ExecutionPolicy Bypass -File ".\run_all.ps1" `
  -CondaEnv $CondaEnv `
  -CondaExe $CondaExe `
  -Seed $Seed `
  -TargetEpisodes $TargetEpisodes `
  -AudioEpochs $AudioEpochs `
  -RadarEpochs $RadarEpochs `
  -GateEpochs $GateEpochs `
  -GateLr $GateLr `
  -GateDropOne $GateDropOne `
  -GateDropTwo $GateDropTwo `
  -GateAuxWeight $GateAuxWeight `
  -FusionMaxConfigs $FusionMaxConfigs `
  -YoloEpochs $YoloEpochs `
  -YoloImgSz $YoloImgSz `
  -YoloFrameStep $YoloFrameStep `
  -YoloRunName $YoloRunName

exit $LASTEXITCODE
