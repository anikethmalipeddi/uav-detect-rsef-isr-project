param(
  [string]$CondaEnv = "uavmm",
  [string]$CondaExe = "$env:USERPROFILE\miniforge3\Scripts\conda.exe",
  [int]$CheckIntervalSec = 300
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

if (-not (Test-Path $CondaExe)) {
  throw "conda executable not found: $CondaExe"
}

New-Item -ItemType Directory -Path "results/logs" -Force | Out-Null

function Write-Status {
  param([string]$Message)
  $ts = (Get-Date).ToString("u")
  $line = "[$ts] $Message"
  Write-Host $line
  Add-Content -Path "results/auto_finish_status.txt" -Value $line
}

function Invoke-Verify {
  & $CondaExe run --no-capture-output -n $CondaEnv python -m src.preprocess.verify_tsms `
    --root data/raw/tsms_drone `
    --out-txt results/tsms_state.txt `
    --out-json results/tsms_state.json `
    --min-total-gb 150.0 `
    --allow-incomplete
  if ($LASTEXITCODE -ne 0) {
    throw "verify_tsms failed with exit code $LASTEXITCODE"
  }
}

function Ensure-DownloaderRunning {
  $running = Get-CimInstance Win32_Process |
    Where-Object { $_.Name -eq "powershell.exe" -and $_.CommandLine -match "download_tsms_full\.ps1" }
  if ($null -eq $running) {
    Write-Status "TSMS downloader not running; starting download_tsms_full.ps1"
    $logOut = "results/tsms_download_auto.log"
    $logErr = "results/tsms_download_auto.err.log"
    Start-Process -FilePath "powershell.exe" `
      -ArgumentList @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", (Join-Path $PSScriptRoot "download_tsms_full.ps1"),
        "-Root", (Resolve-Path "data/raw/tsms_drone"),
        "-Extract",
        "-RetryCount", "3"
      ) `
      -RedirectStandardOutput $logOut `
      -RedirectStandardError $logErr `
      -WindowStyle Hidden | Out-Null
    Start-Sleep -Seconds 3
  }
}

Write-Status "Auto-finish loop started."

while ($true) {
  Ensure-DownloaderRunning
  Invoke-Verify

  $state = Get-Content "results/tsms_state.json" -Raw | ConvertFrom-Json
  $reqDone = [int]$state.required_extracted_ok_count
  $reqTotal = [int]$state.required_entries
  $complete = [bool]$state.complete
  Write-Status ("TSMS progress: required_extracted_ok_count={0}/{1}; complete={2}" -f $reqDone, $reqTotal, $complete)

  if ($complete) {
    Write-Status "TSMS dataset complete. Launching run_all.ps1"
    $runAllLog = "results/logs/run_all_autofinish.log"
    & "powershell.exe" -NoProfile -ExecutionPolicy Bypass -File ".\run_all.ps1" 2>&1 | Tee-Object -FilePath $runAllLog
    if ($LASTEXITCODE -ne 0) {
      Write-Status "run_all.ps1 failed. See $runAllLog"
      exit $LASTEXITCODE
    }
    Write-Status "run_all.ps1 completed successfully."
    break
  }

  Write-Status "Sleeping for $CheckIntervalSec seconds before next TSMS check."
  Start-Sleep -Seconds $CheckIntervalSec
}

Write-Status "Auto-finish loop exited."
