param(
    [string]$Mode = "multi",
    [string]$PythonPath = "C:\Users\jeanj\Documents\School - Research\deepLearning\Scripts\python.exe",
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 5000,
    [string]$StdoutPath = "",
    [string]$StderrPath = "",
    [string]$StatePath = ""
)

$ErrorActionPreference = "Stop"

$launcherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $launcherDir
$serverScriptPath = Join-Path $repoRoot "flask_api_multi.py"
$stdoutPath = if ($StdoutPath) {
    if ([System.IO.Path]::IsPathRooted($StdoutPath)) { $StdoutPath } else { Join-Path $repoRoot $StdoutPath }
} else {
    Join-Path $repoRoot "logs\entity-benchmark-live.out.log"
}
$stderrPath = if ($StderrPath) {
    if ([System.IO.Path]::IsPathRooted($StderrPath)) { $StderrPath } else { Join-Path $repoRoot $StderrPath }
} else {
    Join-Path $repoRoot "logs\entity-benchmark-live.err.log"
}
$statePath = if ($StatePath) {
    if ([System.IO.Path]::IsPathRooted($StatePath)) { $StatePath } else { Join-Path $repoRoot $StatePath }
} else {
    Join-Path $repoRoot "logs\entity-benchmark-live.state.json"
}

$stdoutDir = Split-Path -Parent $stdoutPath
if ($stdoutDir) { New-Item -ItemType Directory -Force -Path $stdoutDir | Out-Null }
$stderrDir = Split-Path -Parent $stderrPath
if ($stderrDir) { New-Item -ItemType Directory -Force -Path $stderrDir | Out-Null }
$stateDir = Split-Path -Parent $statePath
if ($stateDir) { New-Item -ItemType Directory -Force -Path $stateDir | Out-Null }

if (Test-Path $stdoutPath) { Remove-Item $stdoutPath -Force }
if (Test-Path $stderrPath) { Remove-Item $stderrPath -Force }

# Match the baseline server wrapper: launch through cmd.exe so the process keeps
# serving after this helper exits, while stdout/stderr are redirected straight to
# stable log files.
$pythonCommand = "`"$PythonPath`" `"$serverScriptPath`" --mode $Mode --host $BindHost --port $Port"
$cmdCommand = "set PYTHONUNBUFFERED=1 && $pythonCommand 1>> `"$stdoutPath`" 2>> `"$stderrPath`""
$proc = Start-Process `
    -FilePath 'cmd.exe' `
    -ArgumentList @('/d', '/c', $cmdCommand) `
    -WorkingDirectory $repoRoot `
    -PassThru

if (-not $proc) {
    throw "Failed to start entity benchmark server process."
}

$state = @{
    pid = $proc.Id
    host = $BindHost
    port = $Port
    mode = $Mode
    entrypoint = "flask_api_multi.py"
    stdout_path = $stdoutPath
    stderr_path = $stderrPath
    started_at = (Get-Date).ToString("o")
}

$state | ConvertTo-Json -Depth 4 | Set-Content -Path $statePath -Encoding UTF8

Write-Output "started_entity_benchmark_server_pid=$($proc.Id)"
Write-Output "state_file=$statePath"
