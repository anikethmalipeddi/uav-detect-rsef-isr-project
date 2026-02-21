param(
    [ValidateSet("poster", "full")]
    [string]$Mode = "poster"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not (Get-Command streamlit -ErrorAction SilentlyContinue)) {
    Write-Error "streamlit was not found in PATH. Activate your demo environment first."
}

$env:UAV_DEMO_MODE = $Mode
Write-Host "Launching demo in mode: $Mode"
streamlit run demo_app/app.py --server.headless false

