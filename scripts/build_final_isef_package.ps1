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

function Parse-RunTimestampUtc {
  param([string]$MetadataPath)
  if (-not (Test-Path $MetadataPath)) {
    throw "Missing run metadata: $MetadataPath"
  }
  $meta = Get-Content $MetadataPath -Raw | ConvertFrom-Json
  if (-not $meta.timestamp_utc) {
    throw "run_metadata.json missing timestamp_utc."
  }
  return [DateTimeOffset]::Parse($meta.timestamp_utc).UtcDateTime
}

function Assert-NotNewerThan {
  param(
    [string]$Path,
    [datetime]$UtcCutoff
  )
  Require-File $Path
  $item = Get-Item $Path
  $mtime = $item.LastWriteTimeUtc
  if ($mtime -gt $UtcCutoff) {
    throw "Artifact appears newer than final run metadata timestamp: $Path (mtime=$mtime > cutoff=$UtcCutoff)"
  }
}

function To-BoolFromFile {
  param([string]$Path,[string]$Token)
  if (-not (Test-Path $Path)) { return $false }
  $txt = Get-Content $Path -Raw
  return $txt -match [regex]::Escape($Token)
}

$verificationJson = "results/verification_report.json"
Require-File $verificationJson
$verification = Get-Content $verificationJson -Raw | ConvertFrom-Json
if ($verification.overall_status -ne "PASS") {
  throw "Verification report is not PASS. Run scripts/verify_final.ps1 first."
}

$runMetadataPath = "results/run_metadata.json"
$runCutoffUtc = Parse-RunTimestampUtc -MetadataPath $runMetadataPath

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
  "results/fusion_tuning.csv",
  "results/latency_summary.json"
)
foreach($f in $tableFiles){
  Assert-NotNewerThan -Path $f -UtcCutoff $runCutoffUtc
  Copy-Item $f (Join-Path $OutDir "tables") -Force
}

Require-File "results/figures/ablation_comparison.png"
Require-File "results/figures/stress_curves.png"
Require-File "results/figures/missing_modality_comparison.png"
Require-File "results/figures/gate_weight_behavior.png"
Require-File "results/figures/latency_plot.png"
$figureFiles = Get-ChildItem "results/figures" -File -Filter *.png
foreach($fig in $figureFiles) {
  Assert-NotNewerThan -Path $fig.FullName -UtcCutoff $runCutoffUtc
}
Copy-Item "results/figures/*.png" (Join-Path $OutDir "figures") -Force

$verifyFiles = @(
  "results/tsms_state.txt",
  "results/tsms_download_complete.flag",
  "results/dronedetect_state.txt",
  "results/tsms_state.json",
  "results/dronedetect_state.json",
  "results/split_integrity_report.txt",
  "results/split_integrity_report.json",
  "results/label_stats.json",
  "results/label_stats.txt",
  "results/label_per_clip_summary.csv",
  "results/verification_report.txt",
  "results/verification_report.json"
)
foreach($f in $verifyFiles){ Require-File $f; Copy-Item $f (Join-Path $OutDir "verification") -Force }
if (Test-Path "results/label_viz") {
  Get-ChildItem "results/label_viz" -File -Filter *.png | Sort-Object Name | Select-Object -First 20 | ForEach-Object {
    Copy-Item $_.FullName (Join-Path $OutDir "verification") -Force
  }
}

$metaFiles = @(
  "results/run_metadata.json",
  "results/config_used.yaml",
  "results/significance_pr_auc.txt",
  "results/fusion_best_config.json",
  "results/metrics_vision.json",
  "results/fusion_advantage_check.txt"
)
foreach($f in $metaFiles){ Require-File $f; Copy-Item $f (Join-Path $OutDir "metadata") -Force }
if (Test-Path "results/WARNINGS.txt") {
  Copy-Item "results/WARNINGS.txt" (Join-Path $OutDir "metadata") -Force
}
if (Test-Path "results/fallback_warnings.txt") {
  Copy-Item "results/fallback_warnings.txt" (Join-Path $OutDir "metadata/WARNINGS.txt") -Force
}

$fusionBest = Get-Content "results/fusion_best_config.json" -Raw | ConvertFrom-Json
$metricsFusion = Import-Csv "results/metrics_fusion.csv"
$bestFusionRow = $metricsFusion | Sort-Object {[double]$_.pr_auc} -Descending | Select-Object -First 1
$tsmsComplete = To-BoolFromFile -Path "results/tsms_state.txt" -Token "TSMS_DATASET_COMPLETE=TRUE"
$droneComplete = To-BoolFromFile -Path "results/dronedetect_state.txt" -Token "FULL_DRONEDETECT_PRESENT=YES"
$splitPass = To-BoolFromFile -Path "results/split_integrity_report.txt" -Token "SPLIT_INTEGRITY=PASS"
$figureNames = (Get-ChildItem (Join-Path $OutDir "figures") -File -Filter *.png | Sort-Object Name | ForEach-Object { $_.Name })
$figureLines = @()
foreach($name in $figureNames) {
  $figureLines += "- $name"
}
$overlayCount = (Get-ChildItem (Join-Path $OutDir "verification") -File -Filter *.png).Count
$oversize = Get-ChildItem $OutDir -Recurse -File | Where-Object { $_.Length -gt 50MB }

$checklist = @(
  "FINAL ISEF PACKAGE CHECKLIST",
  "",
  "Source of truth: FINAL artifacts copied from results/ generated by the reproducible pipeline.",
  "Verification gate: results/verification_report.json => PASS (required before packaging).",
  "Run metadata cutoff (UTC): $runCutoffUtc",
  "",
  "Dataset verification state:",
  "- TSMS full complete: $tsmsComplete",
  "- DroneDetectionThesis full complete: $droneComplete",
  "- Split integrity PASS: $splitPass",
  "",
  "Best fusion model chosen:",
  "- Tuning-selected config_id: $($fusionBest.selected_config_id)",
  "- Selection metric: $($fusionBest.selection_metric)",
  "- Selected hyperparameters: $($fusionBest.selected_config | ConvertTo-Json -Compress)",
  "- Best TEST PR-AUC row in metrics_fusion.csv: run_id=$($bestFusionRow.run_id), pr_auc=$($bestFusionRow.pr_auc), roc_auc=$($bestFusionRow.roc_auc)",
  "",
  "Included poster figures:"
)
$checklist += $figureLines
$checklist += @(
  "",
  "Directory contents:",
  "- tables/: final metrics/ablation/missing/stress/preds/tuning + latency summary",
  "- figures/: poster-ready PNG plots",
  "- verification/: dataset/split/label stats + verification reports + 20 label overlay PNGs",
  "- metadata/: run metadata, config used, significance, fusion checks, vision metrics",
  "",
  "Raw/processed datasets are NOT included in this folder."
)
if ($oversize.Count -gt 0) {
  $checklist += ""
  $checklist += "WARNING: Some poster-kit files exceed 50 MB:"
  foreach($f in $oversize) {
    $checklist += "- $($f.FullName) ($([Math]::Round($f.Length/1MB,2)) MB)"
  }
  Write-Warning "Poster-kit contains file(s) > 50MB. Review before commit."
}
Set-Content -Path (Join-Path $OutDir "README_CHECKLIST.txt") -Value ($checklist -join "`r`n") -Encoding UTF8

Write-Host "Built $OutDir"
