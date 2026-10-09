param([switch]$SkipModels)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Python = (py -3.11 -c "import sys; print(sys.executable)").Trim()
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & $Python -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --requirement requirements.lock
& .\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
if ($LASTEXITCODE -ne 0) { throw "CPU PyTorch installation failed" }
& .\.venv\Scripts\python.exe -m pip install --requirement requirements-speaker.txt
if ($LASTEXITCODE -ne 0) { throw "SpeechBrain installation failed" }
if (-not $SkipModels) {
    & "$PSScriptRoot\download_models.ps1"
    & .\.venv\Scripts\python.exe scripts\download_ecapa.py
    if ($LASTEXITCODE -ne 0) { throw "ECAPA model download failed" }
}
Write-Host "Setup complete: $Root"
