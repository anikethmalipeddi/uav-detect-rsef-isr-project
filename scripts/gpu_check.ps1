nvidia-smi | Out-File -FilePath results/gpu_check.txt -Encoding utf8
$CondaExe = Join-Path $env:USERPROFILE "miniforge3\\Scripts\\conda.exe"
& $CondaExe run -n uavmm python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)" | Out-File -FilePath results/gpu_check.txt -Append -Encoding utf8
Write-Host "Wrote results/gpu_check.txt"
