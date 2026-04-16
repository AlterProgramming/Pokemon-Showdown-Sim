param(
    [string]$Value,
    [ValidateSet("Auto", "Uuid", "Text")]
    [string]$ValueKind = "Auto",
    [uint16]$CompanyId = 65534,
    [int]$DurationSeconds = 120,
    [switch]$UseCodexSessionId,
    [switch]$DryRun,
    [switch]$PassThru
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementPublisher, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEManufacturerData, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataWriter, Windows, ContentType = WindowsRuntime] | Out-Null

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
        Payload     = $customBytes
        HexPayload  = ([System.BitConverter]::ToString($customBytes) -replace '-', '')
    }
}

$resolvedValue = Resolve-BroadcastValue -ExplicitValue $Value -FromCodexSession:$UseCodexSessionId
$encoded = Encode-BroadcastPayload -InputValue $resolvedValue -RequestedKind $ValueKind

$result = [pscustomobject]@{
    Value           = $encoded.SourceValue
    ValueKind       = $encoded.Kind
    CompanyId       = $CompanyId
    DurationSeconds = $DurationSeconds
    PayloadHex      = $encoded.HexPayload
}

if ($DryRun) {
    if ($PassThru) {
        $result
    } else {
        $result | Format-List
    }
    exit 0
}

$writer = [Windows.Storage.Streams.DataWriter]::new()
$writer.WriteBytes($encoded.Payload)
$buffer = $writer.DetachBuffer()

$advertisement = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisement]::new()
$manufacturerData = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEManufacturerData]::new($CompanyId, $buffer)
[void]$advertisement.ManufacturerData.Add($manufacturerData)

$publisher = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementPublisher]::new($advertisement)
$publisher.Start()
Start-Sleep -Milliseconds 500

if ($publisher.Status.ToString() -ne "Started") {
    $status = $publisher.Status.ToString()
    $publisher.Stop()
    throw "BLE advertisement did not start. Publisher status: $status"
}

try {
    if ($PassThru) {
        $result
    } else {
        Write-Host "Broadcasting BLE payload..."
        $result | Format-List
        Write-Host ""
        Write-Host "Press Ctrl+C to stop early."
    }

    Start-Sleep -Seconds $DurationSeconds
} finally {
    $publisher.Stop()
}
