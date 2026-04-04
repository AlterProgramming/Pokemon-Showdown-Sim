param(
    [string]$Mode = "multi",
    [string]$PythonPath = "C:\Users\jeanj\Documents\School - Research\deepLearning\Scripts\python.exe",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000
)

$ErrorActionPreference = "Stop"

$launcherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $launcherDir
Set-Location $repoRoot

Write-Output "[model-server] python=$PythonPath"
Write-Output "[model-server] mode=$Mode"
Write-Output "[model-server] listening=http://$BindHost`:$Port"

& $PythonPath ".\flask_api_multi.py" `
    --mode $Mode `
    --host $BindHost `
    --port $Port
