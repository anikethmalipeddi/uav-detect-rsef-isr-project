param(
  [string]$EnvName = "uavmm"
)

$CondaExe = Join-Path $env:USERPROFILE "miniforge3\\Scripts\\conda.exe"
if (-not (Test-Path $CondaExe)) {
  throw "Conda executable not found at $CondaExe"
}

Write-Host "Creating conda environment from environment.yml ..."
& $CondaExe env create -f environment.yml
& $CondaExe run -n $EnvName python -c "import sys; print(sys.version)"

Write-Host "Verifying GPU ..."
nvidia-smi
& $CondaExe run -n $EnvName python -c "import torch; print('torch', torch.__version__); print('cuda?', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

Write-Host "Installing git-lfs ..."
git lfs install

Write-Host "Setup complete."
