[CmdletBinding()]
param(
    [string]$ModelName = "entity_action_bc_v1_run1",
    [string]$OutputDir = ".\artifacts",
    [int]$MaxBattles = 5000,
    [int]$Epochs = 30,
    [int]$BatchSize = 256,
    [switch]$MoveOnly,
    [switch]$PredictValue = $true,
    [switch]$PredictTurnOutcome = $true,
    [string[]]$DataPaths = @(),
    [string]$ExtraArgs = ""
)

# Opens a separate visible PowerShell window for entity-family training while
# mirroring stdout/stderr into a log file we can watch from this workspace.
$ErrorActionPreference = "Stop"

$launcherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $launcherDir
$logsDir = Join-Path $repoRoot "logs"
New-Item -ItemType Directory -Force -Path $logsDir | Out-Null

$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logsDir "$ModelName-$timestamp.train.out.log"
$statePath = Join-Path $logsDir "$ModelName-$timestamp.train.state.json"

# Prefer the existing project-local deepLearning environment so the training
# stack stays isolated from the system Python installation.
$venvPython = Join-Path $repoRoot "..\..\deepLearning\Scripts\python.exe"
$venvActivate = Join-Path $repoRoot "..\..\deepLearning\Scripts\Activate.ps1"
$pythonExe = $null
$activateScript = $null
if (Test-Path $venvPython) {
    $pythonExe = (Resolve-Path $venvPython).Path
    if (Test-Path $venvActivate) {
        $activateScript = (Resolve-Path $venvActivate).Path
    }
} else {
    # Fallback is only for cases where the wrapper environment is missing.
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Could not find a usable Python interpreter for the training launcher."
    }
    $pythonExe = $pythonCommand.Source
}

$argList = @(
    "-u"
    ".\train_entity_action.py"
    "--model-name", $ModelName
    "--output-dir", $OutputDir
    "--max-battles", "$MaxBattles"
    "--epochs", "$Epochs"
    "--batch-size", "$BatchSize"
)

if ($PredictValue) {
    $argList += "--predict-value"
}
if ($PredictTurnOutcome) {
    $argList += "--predict-turn-outcome"
}
if ($MoveOnly) {
    $argList += "--move-only"
}
if ($DataPaths.Count -gt 0) {
    $argList += $DataPaths
}
if ($ExtraArgs) {
    $argList += $ExtraArgs
}

# Build the Python command as text so the new PowerShell window can activate
# the environment, run the trainer, and tee the live output to disk.
$pythonCommandText = @(
    "&", ("'" + $pythonExe.Replace("'", "''") + "'")
) + ($argList | ForEach-Object {
    if ($_ -match "^[A-Za-z0-9_./:\\-]+$") {
        $_
    } else {
        "'" + $_.Replace("'", "''") + "'"
    }
})

$trainingCommand = ($pythonCommandText -join " ") + " 2>&1 | Tee-Object -FilePath '" + $logPath.Replace("'", "''") + "'"
$windowTitle = "Entity Training - $ModelName"
$activationPrefix = ""
if ($activateScript) {
    $activationPrefix = "& '" + $activateScript.Replace("'", "''") + "'; "
}
$psArgs = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-Command",
    "`$Host.UI.RawUI.WindowTitle = '$($windowTitle.Replace("'", "''"))'; `$env:PYTHONUNBUFFERED='1'; Set-Location '$($repoRoot.Replace("'", "''"))'; $activationPrefix$trainingCommand"
)

$proc = Start-Process -FilePath "powershell.exe" -ArgumentList $psArgs -WorkingDirectory $repoRoot -PassThru

# The state file is the lightweight contract that lets other scripts or an
# external monitor discover the PID, log path, and artifact location.
$state = [ordered]@{
    model_name = $ModelName
    process_id = $proc.Id
    started_at = (Get-Date).ToString("o")
    repo_root = $repoRoot
    log_path = $logPath
    state_path = $statePath
    output_dir = (Resolve-Path -Path (Join-Path $repoRoot $OutputDir)).Path
    python_executable = $pythonExe
    activate_script = $activateScript
    command_args = $argList
}
$state | ConvertTo-Json -Depth 6 | Set-Content -Path $statePath -Encoding UTF8

Write-Output "started_entity_training_pid=$($proc.Id)"
Write-Output "entity_training_log=$logPath"
Write-Output "entity_training_state=$statePath"
