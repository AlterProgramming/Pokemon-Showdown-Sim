param(
    [string]$Mode = "multi",
    [string]$PythonPath = "",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

function Resolve-PythonExecutable {
    param(
        [string]$RequestedPath,
        [string]$RepoRoot
    )

    if ($RequestedPath) {
        return $RequestedPath
    }
    if ($env:PS_AGENT_PYTHON) {
        return $env:PS_AGENT_PYTHON
    }

    $candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "venv\Scripts\python.exe"),
        (Join-Path $RepoRoot "..\.venv\Scripts\python.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) {
            return (Resolve-Path $candidate).Path
        }
    }

    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        return $pythonCommand.Source
    }

    throw "Could not find a usable Python interpreter. Set PS_AGENT_PYTHON or pass -PythonPath."
}

$launcherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $launcherDir
$PythonPath = Resolve-PythonExecutable -RequestedPath $PythonPath -RepoRoot $repoRoot
Set-Location $repoRoot

Write-Output "[model-server] python=$PythonPath"
Write-Output "[model-server] mode=$Mode"
Write-Output "[model-server] listening=http://$BindHost`:$Port"

& $PythonPath ".\flask_api_multi.py" `
    --mode $Mode `
    --host $BindHost `
    --port $Port
