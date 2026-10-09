$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Models = Join-Path $Root "models"
$Downloads = Join-Path $Root "data\downloads"
New-Item -ItemType Directory -Force -Path $Models, $Downloads | Out-Null

$Items = @(
    @{
        Name = "vosk-model-spk-0.4"
        Url = "https://alphacephei.com/vosk/models/vosk-model-spk-0.4.zip"
    },
    @{
        Name = "vosk-model-small-ru-0.22"
        Url = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
    }
)

foreach ($Item in $Items) {
    $Target = Join-Path $Models $Item.Name
    if (Test-Path $Target) {
        Write-Host "Already present: $($Item.Name)"
        continue
    }
    $Zip = Join-Path $Downloads "$($Item.Name).zip"
    Write-Host "Downloading $($Item.Name)..."
    Invoke-WebRequest -Uri $Item.Url -OutFile $Zip
    Expand-Archive -Path $Zip -DestinationPath $Models -Force
}
Write-Host "Models ready in $Models"

