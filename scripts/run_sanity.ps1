param(
  [string]$CondaEnv = "uavmm",
  [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path $CondaExe)) {
  throw "conda executable not found: $CondaExe"
}

function Invoke-Step {
  param(
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][scriptblock]$Body
  )
  Write-Host "==> $Name"
  & $Body
  if ($LASTEXITCODE -ne 0) {
    throw "Step failed: $Name (exit code $LASTEXITCODE)"
  }
}

Invoke-Step "split_integrity" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.split_integrity `
    --episodes data/processed/episodes/episodes.csv `
    --out-txt results/split_integrity_report.txt `
    --out-json results/split_integrity_report.json
}

Invoke-Step "sanity_checks" {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.sanity_checks `
    --episodes data/processed/episodes/episodes.csv `
    --features data/processed/episodes/features_cache.npz `
    --out-json results/sanity_checks.json `
    --out-txt results/sanity_checks.txt
}

if (Test-Path "results/metrics_fusion.csv") {
  Invoke-Step "check_fusion_advantage" {
    & $CondaExe run --no-capture-output -n $CondaEnv python -m src.eval.check_fusion_advantage `
      --metrics results/metrics_fusion.csv `
      --out-txt results/fusion_advantage_check.txt `
      --metric pr_auc `
      --split test
  }
}

Write-Host "Sanity checks complete."
