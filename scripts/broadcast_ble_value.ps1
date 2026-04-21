param(
    [string]$Value,
    [ValidateSet("Auto", "Uuid", "Text")]
    [string]$ValueKind = "Auto",
    [uint16]$CompanyId = 65534,
    [int]$DurationSeconds = 120,
    [string]$LoopbackJsonPath,
    [string]$BrowserBridgeUrl,
    [switch]$UseCodexSessionId,
    [switch]$DryRun,
    [switch]$PassThru
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime

function Resolve-BroadcastValue {
    param(
        [string]$ExplicitValue,
        [switch]$FromCodexSession
    )

    if ($ExplicitValue) {
        return $ExplicitValue
    }

    if ($FromCodexSession -or -not $ExplicitValue) {
        $helperPath = Join-Path $PSScriptRoot "get_codex_session_id.ps1"
        if (-not (Test-Path -LiteralPath $helperPath)) {
            throw "Missing helper script: $helperPath"
        }

        $sessionId = & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File $helperPath
        if (-not $sessionId) {
            throw "The Codex session ID helper returned no value."
        }

        return ($sessionId | Select-Object -First 1).Trim()
    }

    throw "Provide -Value or use -UseCodexSessionId."
}

function Convert-HexToBytes {
    param([string]$Hex)

    if ($Hex.Length % 2 -ne 0) {
        throw "Hex string length must be even."
    }

    $bytes = New-Object byte[] ($Hex.Length / 2)
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        $bytes[$i] = [Convert]::ToByte($Hex.Substring($i * 2, 2), 16)
    }
    return $bytes
}

function Encode-BroadcastPayload {
    param(
        [string]$InputValue,
        [string]$RequestedKind
    )

    $uuidPattern = '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    $resolvedKind = $RequestedKind

    if ($resolvedKind -eq "Auto") {
        if ($InputValue -match $uuidPattern) {
            $resolvedKind = "Uuid"
        } else {
            $resolvedKind = "Text"
        }
    }

    $header = [System.Text.Encoding]::ASCII.GetBytes("CDX1")
    $kindByte = switch ($resolvedKind) {
        "Uuid" { [byte]1 }
        "Text" { [byte]2 }
        default { throw "Unsupported value kind: $resolvedKind" }
    }

    switch ($resolvedKind) {
        "Uuid" {
            if ($InputValue -notmatch $uuidPattern) {
                throw "UUID payloads must look like 12345678-1234-1234-1234-1234567890ab."
            }

            $valueBytes = Convert-HexToBytes ($InputValue -replace '-', '').ToLowerInvariant()
        }
        "Text" {
            $valueBytes = [System.Text.Encoding]::UTF8.GetBytes($InputValue)
        }
    }

    $maxPayloadBytes = 27
    $customBytes = New-Object byte[] ($header.Length + 1 + $valueBytes.Length)
    [Array]::Copy($header, 0, $customBytes, 0, $header.Length)
    $customBytes[$header.Length] = $kindByte
    [Array]::Copy($valueBytes, 0, $customBytes, $header.Length + 1, $valueBytes.Length)

    if ($customBytes.Length -gt $maxPayloadBytes) {
        throw "Encoded payload is $($customBytes.Length) bytes, but BLE advertisement data must stay at or below $maxPayloadBytes bytes. Use a UUID or a shorter text value."
    }

    return [pscustomobject]@{
        Kind        = $resolvedKind
        SourceValue = $InputValue
        HexPayload  = ([System.BitConverter]::ToString($customBytes) -replace '-', '')
    }
}

function Get-BleHelperBuildSpec {
    $frameworkRoot = 'C:\Windows\Microsoft.NET\Framework64\v4.0.30319'
    $compilerPath = Join-Path $frameworkRoot 'csc.exe'
    $runtimeWindowsPath = Join-Path $frameworkRoot 'System.Runtime.WindowsRuntime.dll'
    $systemRuntimePath = 'C:\Windows\Microsoft.NET\assembly\GAC_MSIL\System.Runtime\v4.0_4.0.0.0__b03f5f7f11d50a3a\System.Runtime.dll'
    $windowsWinMdPath = 'C:\Program Files (x86)\Windows Kits\10\UnionMetadata\10.0.19041.0\Windows.winmd'

    foreach ($requiredPath in @($compilerPath, $runtimeWindowsPath, $systemRuntimePath, $windowsWinMdPath)) {
        if (-not (Test-Path -LiteralPath $requiredPath)) {
            throw "Missing required BLE helper dependency: $requiredPath"
        }
    }

    $helperSourcePath = Join-Path $PSScriptRoot 'BleBroadcastHelper.cs'
    if (-not (Test-Path -LiteralPath $helperSourcePath)) {
        throw "Missing BLE helper source file: $helperSourcePath"
    }

    $helperDir = Join-Path $PSScriptRoot '.bin'
    $helperExePath = Join-Path $helperDir 'BleBroadcastHelper.exe'

    return [pscustomobject]@{
        CompilerPath       = $compilerPath
        RuntimeWindowsPath = $runtimeWindowsPath
        SystemRuntimePath  = $systemRuntimePath
        WindowsWinMdPath   = $windowsWinMdPath
        HelperSourcePath   = $helperSourcePath
        HelperDir          = $helperDir
        HelperExePath      = $helperExePath
    }
}

function Build-BleHelperIfNeeded {
    param([pscustomobject]$BuildSpec)

    $needsBuild = -not (Test-Path -LiteralPath $BuildSpec.HelperExePath)
    if (-not $needsBuild) {
        $needsBuild = (Get-Item -LiteralPath $BuildSpec.HelperSourcePath).LastWriteTimeUtc -gt (Get-Item -LiteralPath $BuildSpec.HelperExePath).LastWriteTimeUtc
    }

    if (-not $needsBuild) {
        return
    }

    New-Item -ItemType Directory -Force -Path $BuildSpec.HelperDir | Out-Null
    $compilerArgs = @(
        '/nologo',
        '/target:exe',
        "/out:$($BuildSpec.HelperExePath)",
        "/reference:$($BuildSpec.RuntimeWindowsPath)",
        "/reference:$($BuildSpec.SystemRuntimePath)",
        "/reference:$($BuildSpec.WindowsWinMdPath)",
        $BuildSpec.HelperSourcePath
    )

    & $BuildSpec.CompilerPath @compilerArgs
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $BuildSpec.HelperExePath)) {
        throw "Failed to build the BLE broadcast helper."
    }
}

function Send-BrowserBridgeMessage {
    param(
        [string]$BridgeUrl,
        [pscustomobject]$Message
    )

    if (-not $BridgeUrl) {
        return
    }

    try {
        $body = $Message | ConvertTo-Json -Depth 6
        Invoke-RestMethod -Method Post -Uri $BridgeUrl -ContentType "application/json" -Body $body | Out-Null
    } catch {
        Write-Warning "Failed to publish browser bridge message to ${BridgeUrl}: $($_.Exception.Message)"
    }
}

$resolvedValue = Resolve-BroadcastValue -ExplicitValue $Value -FromCodexSession:$UseCodexSessionId
$encoded = Encode-BroadcastPayload -InputValue $resolvedValue -RequestedKind $ValueKind
$timestampUtc = [DateTimeOffset]::UtcNow.ToString("o")

$result = [pscustomobject]@{
    SessionId       = $encoded.SourceValue
    Value           = $encoded.SourceValue
    ValueKind       = $encoded.Kind
    CompanyId       = $CompanyId
    DurationSeconds = $DurationSeconds
    PredictionScore = 1.0
    PayloadHex      = $encoded.HexPayload
    TimestampUtc    = $timestampUtc
}

if ($DryRun) {
    if ($PassThru) {
        $result
    } else {
        $result | Format-List
    }
    exit 0
}

$loopbackPath = $null
if ($LoopbackJsonPath) {
    if ([System.IO.Path]::IsPathRooted($LoopbackJsonPath)) {
        $loopbackPath = [System.IO.Path]::GetFullPath($LoopbackJsonPath)
    } else {
        $loopbackPath = [System.IO.Path]::GetFullPath((Join-Path $PWD $LoopbackJsonPath))
    }
    $loopbackDir = Split-Path -Parent $loopbackPath
    if ($loopbackDir) {
        New-Item -ItemType Directory -Force -Path $loopbackDir | Out-Null
    }

    $loopbackRecord = [pscustomobject]@{
        SessionId       = $result.SessionId
        Value           = $result.Value
        ValueKind       = $result.ValueKind
        CompanyId       = $result.CompanyId
        DurationSeconds = $result.DurationSeconds
        PredictionScore = $result.PredictionScore
        PayloadHex      = $result.PayloadHex
        TimestampUtc    = $timestampUtc
        WrittenAtUtc    = $timestampUtc
    }

    $loopbackRecord | ConvertTo-Json -Depth 4 | Set-Content -Path $loopbackPath -Encoding UTF8
}

Send-BrowserBridgeMessage -BridgeUrl $BrowserBridgeUrl -Message ([pscustomobject]@{
    source          = "codex"
    target          = "browser"
    kind            = "codex_session_id"
    session_id      = $result.SessionId
    value           = $result.Value
    value_kind      = $result.ValueKind
    company_id      = $result.CompanyId
    duration_seconds = $result.DurationSeconds
    prediction_score = $result.PredictionScore
    payload_hex     = $result.PayloadHex
    timestamp_utc   = $result.TimestampUtc
})

$buildSpec = Get-BleHelperBuildSpec
Build-BleHelperIfNeeded -BuildSpec $buildSpec
$helperArgs = @(
    '--company-id', "$CompanyId",
    '--payload-hex', $encoded.HexPayload,
    '--duration-ms', "$($DurationSeconds * 1000)"
)

try {
    if ($PassThru) {
        $result
    } else {
        Write-Host "Broadcasting BLE payload..."
        $result | Format-List
        if ($loopbackPath) {
            Write-Host "Loopback JSON: $loopbackPath"
        }
        Write-Host ""
        Write-Host "Press Ctrl+C to stop early."
    }

    & $buildSpec.HelperExePath @helperArgs
    if ($LASTEXITCODE -ne 0) {
        throw "The BLE broadcast helper exited with code $LASTEXITCODE."
    }
} finally {}
