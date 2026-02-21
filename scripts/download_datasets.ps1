param(
  [string]$Root = "data/raw",
  [switch]$DownloadTSMSSampleCode
)

$thesis = Join-Path $Root "dronedetect_thesis"
$tsms = Join-Path $Root "tsms_drone"

New-Item -ItemType Directory -Path $Root -Force | Out-Null

if (-not (Test-Path $thesis)) {
  Write-Host "Cloning DroneDetectionThesis dataset ..."
  git clone https://github.com/DroneDetectionThesis/Drone-detection-dataset $thesis
} else {
  Write-Host "DroneDetectionThesis already exists at $thesis"
}

if (-not (Test-Path $tsms)) {
  New-Item -ItemType Directory -Path $tsms -Force | Out-Null
  Write-Host "TSMS folder created at $tsms"
} else {
  Write-Host "TSMS folder already exists at $tsms"
}

Write-Host "Fetching TSMS figshare file catalog ..."
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
$rows | Sort-Object folder,name,id | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $tsms "figshare_files.csv")
Write-Host "Saved TSMS catalog to $tsms\\figshare_files.csv"

if ($DownloadTSMSSampleCode) {
  $dst = Join-Path $tsms "sample_code"
  New-Item -ItemType Directory -Path $dst -Force | Out-Null
  $small = $api.files | Where-Object { $_.size -lt 5000000 }
  foreach($f in $small){
    $name = [IO.Path]::GetFileName($f.name)
    $out = Join-Path $dst $name
    Invoke-WebRequest -Uri $f.download_url -OutFile $out
  }
  Write-Host "Downloaded TSMS sample-code files to $dst"
}
