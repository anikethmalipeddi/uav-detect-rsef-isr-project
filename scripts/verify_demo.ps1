$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$pythonCmd = $null
$candidates = @(
    "python",
    "$HOME\miniforge3\python.exe",
    "C:\Users\Aniketh\miniforge3\python.exe"
)
foreach ($candidate in $candidates) {
    try {
        & $candidate --version *> $null
        if ($LASTEXITCODE -eq 0) {
            $pythonCmd = $candidate
            break
        }
    } catch {
        continue
    }
}

if (-not $pythonCmd) {
    Write-Error "Python not found. Install Python/Conda and re-run."
}

& $pythonCmd tools/verify_demo_cross_platform.py
exit $LASTEXITCODE
