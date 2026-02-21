param(
  [string]$Root = "data/raw/tsms_drone",
  [int]$MaxFiles = 0,
  [int]$RetryCount = 2,
  [int]$BackoffBaseSec = 15,
  [int]$MinExtractedFiles = 1000,
  [string]$CatalogCsv = "",
  [switch]$Extract,
  [switch]$DeleteArchive,
  [string]$CompleteFlag = "results/tsms_download_complete.flag"
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
  $PSNativeCommandUseErrorActionPreference = $false
}
New-Item -ItemType Directory -Path $Root -Force | Out-Null

$sevenZip = "${env:ProgramFiles}\7-Zip\7z.exe"
if ($Extract -and -not (Test-Path $sevenZip)) {
  throw "7z.exe not found at $sevenZip"
}

function Invoke-7zExtractWithFallback {
  param(
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [Parameter(Mandatory = $true)][string]$ExtractDir,
    [Parameter(Mandatory = $true)][string]$SevenZipExe,
    [Parameter(Mandatory = $true)][string]$Tag
  )

  & $SevenZipExe x "-o$ExtractDir" -y $ArchivePath
  if ($LASTEXITCODE -eq 0) {
    return $true
  }

  Write-Warning "Primary extraction failed for $ArchivePath. Trying short-path extraction fallback."
  $tmpRoot = Join-Path $env:TEMP ("tsms_extract_" + $Tag)
  if (Test-Path $tmpRoot) {
    Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
  }
  New-Item -ItemType Directory -Path $tmpRoot -Force | Out-Null

  & $SevenZipExe x "-o$tmpRoot" -y $ArchivePath
  $extractCode = $LASTEXITCODE
  if ($extractCode -ne 0) {
    Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
    return $false
  }

  New-Item -ItemType Directory -Path $ExtractDir -Force | Out-Null
  & robocopy.exe $tmpRoot $ExtractDir /E /NFL /NDL /NJH /NJS /NP *> $null
  $copyCode = $LASTEXITCODE
  Remove-Item -Recurse -Force $tmpRoot -ErrorAction SilentlyContinue
  if ($copyCode -le 7) {
    return $true
  }

  Write-Warning "Short-path extraction copy failed (robocopy exit code $copyCode)."
  return $false
}

$catalog = if ([string]::IsNullOrWhiteSpace($CatalogCsv)) { Join-Path $Root "figshare_files.csv" } else { $CatalogCsv }
if (-not (Test-Path $catalog)) {
  if (-not [string]::IsNullOrWhiteSpace($CatalogCsv)) {
    throw "Catalog not found: $catalog"
  }
  Write-Host "Catalog not found. Fetching from figshare API..."
  $api = Invoke-RestMethod -Uri "https://api.figshare.com/v2/articles/30027313" -Method Get
  $rows = foreach($f in $api.files){
    [PSCustomObject]@{
      id = $f.id
      name = $f.name
      size_bytes = $f.size
      size_gb = [math]::Round($f.size/1GB,3)
      download_url = $f.download_url
      folder = $api.folder_structure.($f.id.ToString())
    }
  }
  $rows | Sort-Object folder,name,id | Export-Csv -NoTypeInformation -Encoding UTF8 $catalog
}

$rows = Import-Csv $catalog
if ($MaxFiles -gt 0) {
  $rows = $rows | Select-Object -First $MaxFiles
}

Write-Host "Downloading $($rows.Count) TSMS files..."
foreach($r in $rows){
  $sub = if ([string]::IsNullOrWhiteSpace($r.folder)) { "" } else { $r.folder.Replace('/','\') }
  $dstDir = Join-Path $Root $sub
  New-Item -ItemType Directory -Path $dstDir -Force | Out-Null

  $base = [IO.Path]::GetFileNameWithoutExtension($r.name)
  $ext = [IO.Path]::GetExtension($r.name)
  $file = "{0}_{1}{2}" -f $base, $r.id, $ext
  $out = Join-Path $dstDir $file
  $isArchive = $out.ToLower().EndsWith(".7z")
  $extractDir = [IO.Path]::Combine($dstDir, [IO.Path]::GetFileNameWithoutExtension($file))

  if ($Extract -and $isArchive -and (Test-Path $extractDir)) {
    $extractedSample = Get-ChildItem $extractDir -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First $MinExtractedFiles
    $extractLooksComplete = ($extractedSample.Count -ge $MinExtractedFiles)
    if ($extractLooksComplete) {
      if ($DeleteArchive) {
        Write-Host "Skip (already extracted): $file"
        continue
      }
      if (Test-Path $out) {
        $have = (Get-Item $out).Length
        if ($have -eq [int64]$r.size_bytes) {
          Write-Host "Skip (downloaded+extracted): $file"
          continue
        }
      }
    }
  }

  if (-not $Extract -and (Test-Path $out)) {
    $have = (Get-Item $out).Length
    if ($have -eq [int64]$r.size_bytes) {
      Write-Host "Skip (exists): $file"
      continue
    }
  }

  $ok = $false
  for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
    $needDownload = $true
    if (Test-Path $out) {
      $have = (Get-Item $out).Length
      if ($have -eq [int64]$r.size_bytes) {
        $needDownload = $false
      } else {
        Write-Host "Removing partial file before retry: $file"
        Remove-Item -Force $out -ErrorAction SilentlyContinue
      }
    }

    if ($needDownload) {
      Write-Host "Downloading (attempt $attempt/$RetryCount): $file ($($r.size_gb) GB)"
      # Resume enabled with -C -
      & curl.exe -L -C - -o $out $r.download_url
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "Download failed for $file on attempt $attempt."
        if ($attempt -lt $RetryCount) {
          $sleepSec = [Math]::Max(1, [Math]::Pow(2, $attempt - 1) * $BackoffBaseSec)
          Write-Host "Backoff sleep $sleepSec sec before retry."
          Start-Sleep -Seconds $sleepSec
        }
        continue
      }
      $downloadSize = (Get-Item $out).Length
      if ($downloadSize -ne [int64]$r.size_bytes) {
        Write-Warning "Downloaded size mismatch for $file on attempt $attempt. Expected=$($r.size_bytes) got=$downloadSize"
        Remove-Item -Force $out -ErrorAction SilentlyContinue
        if ($attempt -lt $RetryCount) {
          $sleepSec = [Math]::Max(1, [Math]::Pow(2, $attempt - 1) * $BackoffBaseSec)
          Write-Host "Backoff sleep $sleepSec sec before retry."
          Start-Sleep -Seconds $sleepSec
        }
        continue
      }
    }

    if ($Extract -and $isArchive) {
      New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
      & $sevenZip t $out *> $null
      if ($LASTEXITCODE -ne 0) {
        Write-Warning "Archive validation failed for $file on attempt $attempt. Re-downloading."
        Remove-Item -Force $out -ErrorAction SilentlyContinue
        if ($attempt -lt $RetryCount) {
          $sleepSec = [Math]::Max(1, [Math]::Pow(2, $attempt - 1) * $BackoffBaseSec)
          Write-Host "Backoff sleep $sleepSec sec before retry."
          Start-Sleep -Seconds $sleepSec
        }
        continue
      }
      Write-Host "Extracting: $file -> $extractDir"
      $extractOk = Invoke-7zExtractWithFallback -ArchivePath $out -ExtractDir $extractDir -SevenZipExe $sevenZip -Tag "$($r.id)"
      if (-not $extractOk) {
        Write-Warning "Extraction failed for $file on attempt $attempt. Re-downloading."
        Remove-Item -Recurse -Force $extractDir -ErrorAction SilentlyContinue
        Remove-Item -Force $out -ErrorAction SilentlyContinue
        if ($attempt -lt $RetryCount) {
          $sleepSec = [Math]::Max(1, [Math]::Pow(2, $attempt - 1) * $BackoffBaseSec)
          Write-Host "Backoff sleep $sleepSec sec before retry."
          Start-Sleep -Seconds $sleepSec
        }
        continue
      }
      if ($DeleteArchive) {
        Remove-Item -Force $out -ErrorAction SilentlyContinue
      }
    }

    $ok = $true
    break
  }

  if (-not $ok) {
    Write-Warning "Failed after $RetryCount attempts: $file"
  }
}

Write-Host "TSMS download loop complete."

# Final completion check for required dataset archives only.
$required = $rows | Where-Object {
  ($_.name.ToLower().EndsWith(".7z")) -and ($_.folder -like "*TSMS-Drone/Dataset*")
}
$okCount = 0
$badCount = 0
$missing = @()
foreach($r in $required){
  $sub = if ([string]::IsNullOrWhiteSpace($r.folder)) { "" } else { $r.folder.Replace('/','\') }
  $dstDir = Join-Path $Root $sub
  $base = [IO.Path]::GetFileNameWithoutExtension($r.name)
  $ext = [IO.Path]::GetExtension($r.name)
  $archiveName = "{0}_{1}{2}" -f $base, $r.id, $ext
  $archivePath = Join-Path $dstDir $archiveName
  $extractDir = [IO.Path]::Combine($dstDir, [IO.Path]::GetFileNameWithoutExtension($archiveName))
  $sizeOk = (Test-Path $archivePath) -and ((Get-Item $archivePath).Length -eq [int64]$r.size_bytes)
  $extractSample = if (Test-Path $extractDir) { Get-ChildItem $extractDir -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First $MinExtractedFiles } else { @() }
  $extractOk = (Test-Path $extractDir) -and ($extractSample.Count -ge $MinExtractedFiles)
  $entryOk = if ($DeleteArchive) { $extractOk } else { ($sizeOk -and $extractOk) }
  if ($entryOk) {
    $okCount += 1
  } else {
    $badCount += 1
    if ($missing.Count -lt 30) {
      $missing += [PSCustomObject]@{ id=$r.id; name=$r.name; archive=$archivePath; extract=$extractDir }
    }
  }
}

if ($badCount -eq 0 -and $required.Count -gt 0) {
  $flagDir = Split-Path -Parent $CompleteFlag
  if ($flagDir -and -not (Test-Path $flagDir)) {
    New-Item -ItemType Directory -Path $flagDir -Force | Out-Null
  }
  Set-Content -Path $CompleteFlag -Value "TSMS_DOWNLOAD_COMPLETE=TRUE" -Encoding UTF8
  Write-Host "TSMS_DOWNLOAD_COMPLETE=TRUE"
  Write-Host "COMPLETE"
} else {
  if (Test-Path $CompleteFlag) {
    Remove-Item -Force $CompleteFlag -ErrorAction SilentlyContinue
  }
  Write-Host "TSMS_DOWNLOAD_COMPLETE=FALSE"
  Write-Host "Required archives verified: $okCount/$($required.Count)"
  if ($missing.Count -gt 0) {
    Write-Host "Missing or invalid required entries (first 30):"
    foreach($m in $missing){
      Write-Host "- id=$($m.id) name=$($m.name) archive=$($m.archive) extract=$($m.extract)"
    }
  }
}
