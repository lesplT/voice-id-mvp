$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
& "$PSScriptRoot\generate_synthetic_wav.ps1"
& .\.venv\Scripts\python.exe "$PSScriptRoot\smoke_test.py"
if ($LASTEXITCODE -ne 0) { throw "Smoke test failed" }

