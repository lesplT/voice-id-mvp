$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "Virtual environment is missing. Run scripts\setup.ps1 first."
}
& $Python -m voice_id_mvp.benchmark.desktop
