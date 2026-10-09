$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment is missing. Run scripts\setup.ps1 first."
}

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

$QdrantMode = "local"
$DockerAvailable = $false
try {
    docker info 2>$null | Out-Null
    $DockerAvailable = ($LASTEXITCODE -eq 0)
} catch {
    $DockerAvailable = $false
}
# Do not silently switch an existing embedded voice database after Docker starts.
if (Test-Path "data\qdrant_local\meta.json") {
    $DockerAvailable = $false
    Write-Host "Preserving existing embedded voice database for native Windows services."
}
if ($DockerAvailable) {
    docker compose up -d qdrant
    $QdrantMode = "docker"
} else {
    # PowerShell removes an environment variable when it is set to an empty
    # string, so use an explicit sentinel that the repository understands.
    $env:QDRANT_URL = "local"
    Write-Warning "Docker engine is unavailable; using serialized qdrant-client local mode."
}

New-Item -ItemType Directory -Force -Path "data\logs", "data\run" | Out-Null
$Services = @(
    @{ Name="workspace"; Module="voice_id_mvp.services.workspace_app:app"; Port=8000 },
    @{ Name="enrollment"; Module="voice_id_mvp.services.enrollment_app:app"; Port=8001 },
    @{ Name="identification"; Module="voice_id_mvp.services.identification_app:app"; Port=8002 },
    @{ Name="transcription"; Module="voice_id_mvp.services.transcription_app:app"; Port=8003 }
)
$Pids = @{}
foreach ($Service in $Services) {
    $Listener = Get-NetTCPConnection -LocalPort $Service.Port -State Listen -ErrorAction SilentlyContinue
    if ($Listener) {
        throw "Port $($Service.Port) is already in use. Stop existing project services before starting again."
    }
}
foreach ($Service in $Services) {
    $Out = Join-Path $Root "data\logs\$($Service.Name).out.log"
    $Err = Join-Path $Root "data\logs\$($Service.Name).err.log"
    $Process = Start-Process -FilePath "$Root\.venv\Scripts\python.exe" -ArgumentList @(
        "-m", "uvicorn", $Service.Module, "--host", "127.0.0.1", "--port", $Service.Port
    ) -WorkingDirectory $Root -WindowStyle Hidden -RedirectStandardOutput $Out -RedirectStandardError $Err -PassThru
    $Pids[$Service.Name] = $Process.Id
}
$Pids["qdrant_mode"] = $QdrantMode
$Pids | ConvertTo-Json | Set-Content -Encoding UTF8 "data\run\services.json"

$Deadline = (Get-Date).AddSeconds(90)
foreach ($Service in $Services) {
    do {
        Start-Sleep -Milliseconds 500
        try {
            $Health = Invoke-RestMethod "http://127.0.0.1:$($Service.Port)/health" -TimeoutSec 2
            break
        } catch {
            if ((Get-Date) -gt $Deadline) {
                throw "$($Service.Name) did not become healthy. Check data\logs."
            }
        }
    } while ($true)
    Write-Host "$($Service.Name): http://127.0.0.1:$($Service.Port) - OK"
}
Write-Host "Qdrant mode: $QdrantMode"
Write-Host "Open the workspace: http://127.0.0.1:8000"
