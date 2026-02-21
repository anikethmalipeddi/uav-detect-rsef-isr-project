param(
  [string]$OutDir = "final_isef_package"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Require-File {
  param([string]$Path)
  if (-not (Test-Path $Path)) {
    throw "Required artifact missing: $Path"
  }
}

function Copy-IfExists {
  param([string]$Src,[string]$DstDir)
  if (Test-Path $Src) {
    New-Item -ItemType Directory -Path $DstDir -Force | Out-Null
    Copy-Item -Path $Src -Destination $DstDir -Force
  }
}

if (Test-Path $OutDir) {
  Remove-Item -Recurse -Force $OutDir
}
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutDir "tables") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutDir "figures") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutDir "verification") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $OutDir "metadata") -Force | Out-Null

$tableFiles = @(
  "results/metrics_unimodal.csv",
  "results/metrics_fusion.csv",
  "results/ablation_table.csv",
  "results/missing_modality.csv",
  "results/stress_audio_snr.csv",
  "results/stress_vision_corruptions.csv",
  "results/stress_radar_clutter.csv",
  "results/preds_fusion.csv",
  "results/fusion_tuning.csv"
)
foreach($f in $tableFiles){ Require-File $f; Copy-Item $f (Join-Path $OutDir "tables") -Force }

Require-File "results/figures/ablation_comparison.png"
Require-File "results/figures/stress_curves.png"
Require-File "results/figures/missing_modality_comparison.png"
Require-File "results/figures/gate_weight_behavior.png"
Require-File "results/figures/latency_plot.png"
Copy-Item "results/figures/*.png" (Join-Path $OutDir "figures") -Force

$verifyFiles = @(
  "results/tsms_state.txt",
  "results/tsms_download_complete.flag",
  "results/dronedetect_state.txt",
  "results/tsms_state.json",
  "results/dronedetect_state.json",
  "results/split_integrity_report.txt",
  "results/label_stats.txt",
  "results/label_per_clip_summary.csv"
)
foreach($f in $verifyFiles){ Require-File $f; Copy-Item $f (Join-Path $OutDir "verification") -Force }
if (Test-Path "results/label_viz") {
  Get-ChildItem "results/label_viz" -File | Select-Object -First 20 | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $OutDir "verification") -Force
  }
}

$metaFiles = @(
  "results/run_metadata.json",
  "results/config_used.yaml",
  "results/significance_pr_auc.txt",
  "results/latency_summary.json",
  "results/fusion_advantage_check.txt"
)
foreach($f in $metaFiles){ Require-File $f; Copy-Item $f (Join-Path $OutDir "metadata") -Force }
if (Test-Path "results/WARNINGS.txt") {
  Copy-Item "results/WARNINGS.txt" (Join-Path $OutDir "metadata") -Force
}
if (Test-Path "results/fallback_warnings.txt") {
  Copy-Item "results/fallback_warnings.txt" (Join-Path $OutDir "metadata/WARNINGS.txt") -Force
}

$checklist = @(
  "FINAL ISEF PACKAGE CHECKLIST",
  "tables/: metrics, ablations, missing-modality, stress, preds, tuning",
  "figures/: ablation, stress, missing-modality, gate-behavior, latency",
  "verification/: TSMS state, DroneDetection state, split integrity, label stats, label viz samples",
  "metadata/: run metadata, config used, significance, latency summary, fusion advantage check",
  "No raw/processed datasets included."
)
Set-Content -Path (Join-Path $OutDir "README_CHECKLIST.txt") -Value ($checklist -join "`r`n") -Encoding UTF8

Write-Host "Built $OutDir"
