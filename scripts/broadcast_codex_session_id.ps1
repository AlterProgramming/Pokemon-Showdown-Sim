param(
    [int]$DurationSeconds = 120,
    [uint16]$CompanyId = 65534,
    [string]$LoopbackJsonPath,
    [string]$BrowserBridgeUrl,
    [switch]$DryRun,
    [switch]$PassThru
)

$ErrorActionPreference = "Stop"

$broadcastScript = Join-Path $PSScriptRoot "broadcast_ble_value.ps1"
if (-not (Test-Path -LiteralPath $broadcastScript)) {
    throw "Missing BLE broadcast script: $broadcastScript"
}

$args = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $broadcastScript,
    "-UseCodexSessionId",
    "-DurationSeconds", "$DurationSeconds",
    "-CompanyId", "$CompanyId"
)

if ($LoopbackJsonPath) {
    $args += @("-LoopbackJsonPath", $LoopbackJsonPath)
}

if ($BrowserBridgeUrl) {
    $args += @("-BrowserBridgeUrl", $BrowserBridgeUrl)
}

if ($DryRun) {
    $args += "-DryRun"
}

if ($PassThru) {
    $args += "-PassThru"
}

& "$PSHOME\powershell.exe" @args
