param(
  [string]$CondaEnv = "uavmm",
  [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe"
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path $CondaExe)) {
  throw "conda executable not found: $CondaExe"
}

& $CondaExe run --no-capture-output -n $CondaEnv python tools/verify_final_run.py
exit $LASTEXITCODE
