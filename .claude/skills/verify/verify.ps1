# Usage (from anywhere): powershell -File .claude/skills/verify/verify.ps1 launch|doctor|stop [-Port 8765] [-Full]
param(
    [Parameter(Mandatory)][ValidateSet('launch', 'doctor', 'stop')][string]$Action,
    [int]$Port = 8765,
    [switch]$Full
)
$root = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$evidence = "$PSScriptRoot\evidence"
$pidFile = "$evidence\server.pid"
$base = "http://127.0.0.1:$Port"

function Get-OwnedPid { if (Test-Path $pidFile) { [int](Get-Content $pidFile) } }

# The venv python.exe is a launcher; the process that owns the port is its child.
function Test-Descendant([int]$ChildId, [int]$AncestorId) {
    while ($ChildId) {
        if ($ChildId -eq $AncestorId) { return $true }
        $ChildId = (Get-CimInstance Win32_Process -Filter "ProcessId=$ChildId").ParentProcessId
    }
    $false
}

switch ($Action) {
    'launch' {
        if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
            throw "Port $Port is already in use. Not starting a second instance; pick -Port or stop the owner."
        }
        New-Item -ItemType Directory -Force $evidence | Out-Null
        Set-Location $root
        # Verification deployment drops ask_genome (its import of src.insight_generation.roi is broken).
        # -Full uses the real deployments/local.yaml.
        $env:DEPLOYMENT_PATH = if ($Full) { 'deployments/local.yaml' } else { '.claude/skills/verify/deployment.verify.yaml' }
        $proc = Start-Process -FilePath "$root\.venv\Scripts\python.exe" -WindowStyle Hidden -PassThru `
            -ArgumentList '-W', 'ignore', '-m', 'uvicorn', 'agentic_orchestration.main:app', '--host', '127.0.0.1', '--port', $Port `
            -RedirectStandardOutput "$evidence\server.log" -RedirectStandardError "$evidence\server.err.log"
        $proc.Id | Set-Content $pidFile
        for ($i = 0; $i -lt 30; $i++) {
            if ($proc.HasExited) { Get-Content "$evidence\server.err.log" -Tail 8; throw 'Server exited during startup.' }
            try { Invoke-RestMethod "$base/health" -TimeoutSec 3 | Out-Null; "ready: $base (pid $($proc.Id))"; return } catch { Start-Sleep 3 }
        }
        throw "Not ready after 90s; see $evidence\server.err.log"
    }
    'doctor' {
        $owned = Get-OwnedPid
        if (-not $owned -or -not (Get-Process -Id $owned -ErrorAction SilentlyContinue)) { throw 'No launched instance (pid file missing or process dead). Run launch.' }
        $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $listener -or -not (Test-Descendant $listener.OwningProcess $owned)) { throw "Port $Port is not owned by our pid $owned." }
        $build = Invoke-RestMethod "$base/build"
        "ok: pid $owned owns :$Port, health=$((Invoke-RestMethod "$base/health").status), deployment=$($build.deployment.name) v$($build.application_version)"
    }
    'stop' {
        $owned = Get-OwnedPid
        if ($owned -and (Get-Process -Id $owned -ErrorAction SilentlyContinue)) { taskkill /PID $owned /T /F | Out-Null; "stopped pid $owned and children" } else { 'nothing to stop' }
        Remove-Item $pidFile -ErrorAction SilentlyContinue
    }
}
