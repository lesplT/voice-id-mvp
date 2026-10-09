$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$State = "data\run\services.json"
if (Test-Path $State) {
    $Pids = Get-Content $State -Raw | ConvertFrom-Json
    $Snapshot = Get-CimInstance Win32_Process
    foreach ($Name in "workspace", "enrollment", "identification", "transcription") {
        $Id = $Pids.$Name
        $Module = "voice_id_mvp.services.$($Name)_app:app"
        $Parent = $Snapshot | Where-Object {
            $_.ProcessId -eq $Id -and $_.CommandLine -like "*$Root\.venv\Scripts\python.exe*" -and
            $_.CommandLine -like "*-m uvicorn $Module *"
        }
        if ($Parent) {
            # Windows venv python.exe may launch a child interpreter. Stop the
            # validated children first, otherwise they can retain the ports.
            $Children = $Snapshot | Where-Object {
                $_.ParentProcessId -eq $Id -and $_.Name -eq "python.exe" -and
                $_.CommandLine -like "*-m uvicorn $Module *"
            }
            foreach ($Child in $Children) {
                Stop-Process -Id $Child.ProcessId -ErrorAction SilentlyContinue
            }
            Stop-Process -Id $Id -ErrorAction SilentlyContinue
            Write-Host "Stopped $Name ($Id and verified children)"
        } elseif ($Id) {
            Write-Warning "Skipping $Name PID ${Id}: process ownership could not be verified."
        }
    }
}
docker compose stop qdrant *> $null
