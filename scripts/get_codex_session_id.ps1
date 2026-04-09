param(
    [string]$LogsRoot = "$env:LOCALAPPDATA\Codex\Logs",
    [switch]$AppRunId
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LogsRoot)) {
    throw "Codex log directory not found: $LogsRoot"
}

$logs = Get-ChildItem -Path $LogsRoot -Recurse -File -Filter "codex-desktop-*.log" |
    Sort-Object LastWriteTime -Descending

if (-not $logs) {
    throw "No Codex desktop logs found under: $LogsRoot"
}

if ($AppRunId) {
    $latestLog = $logs | Select-Object -First 1
    if ($latestLog.BaseName -match '^codex-desktop-([0-9a-f-]+)-\d+-t\d+-i\d+-.*$') {
        $matches[1]
        exit 0
    }

    throw "Could not parse the app run ID from log name: $($latestLog.Name)"
}

foreach ($log in $logs) {
    $match = Select-String -Path $log.FullName -Pattern 'conversationId=([0-9a-f-]+)' |
        Select-Object -Last 1

    if ($match -and $match.Matches.Count -gt 0) {
        $match.Matches[0].Groups[1].Value
        exit 0
    }
}

throw "Could not find a conversation ID in the recent Codex desktop logs."
