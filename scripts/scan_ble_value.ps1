param(
    [uint16]$CompanyId = 65534,
    [int]$TimeoutSeconds = 30,
    [string]$LoopbackJsonPath,
    [switch]$FirstOnly,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEScanningMode, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows, ContentType = WindowsRuntime] | Out-Null

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

function Convert-BytesToUuidString {
    param([byte[]]$Bytes)

    if ($Bytes.Length -ne 16) {
        throw "UUID payloads must be exactly 16 bytes."
    }

    $hex = [System.BitConverter]::ToString($Bytes).Replace('-', '').ToLowerInvariant()
    return "{0}-{1}-{2}-{3}-{4}" -f $hex.Substring(0, 8), $hex.Substring(8, 4), $hex.Substring(12, 4), $hex.Substring(16, 4), $hex.Substring(20, 12)
}

function Convert-LoopbackTimestamp {
    param([object]$Value)

    if ($null -eq $Value) {
        return [DateTimeOffset]::UtcNow
    }

    if ($Value -is [DateTimeOffset]) {
        return $Value
    }

    if ($Value -is [DateTime]) {
        return [DateTimeOffset]::new($Value.ToUniversalTime())
    }

    $text = [string]$Value
    if ($text -match '^/Date\((\d+)\)/$') {
        return [DateTimeOffset]::FromUnixTimeMilliseconds([int64]$matches[1])
    }

    try {
        return [DateTimeOffset]::Parse($text)
    } catch {
        return [DateTimeOffset]::UtcNow
    }
}

function Decode-BroadcastPayload {
    param([byte[]]$Bytes)

    if ($Bytes.Length -lt 5) {
        return $null
    }

    $magic = [System.Text.Encoding]::ASCII.GetString($Bytes, 0, 4)
    if ($magic -ne "CDX1") {
        return $null
    }

    $kind = $Bytes[4]
    $payloadLength = $Bytes.Length - 5
    $payload = New-Object byte[] $payloadLength
    [Array]::Copy($Bytes, 5, $payload, 0, $payloadLength)

    switch ($kind) {
        1 {
            return [pscustomobject]@{
                ValueKind = "Uuid"
                Value     = Convert-BytesToUuidString $payload
            }
        }
        2 {
            return [pscustomobject]@{
                ValueKind = "Text"
                Value     = [System.Text.Encoding]::UTF8.GetString($payload)
            }
        }
        default {
            return [pscustomobject]@{
                ValueKind = "Unknown"
                Value     = [System.BitConverter]::ToString($payload).Replace('-', '')
            }
        }
    }
}

$ReadLoopbackPayload = {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }

    $json = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    if (-not $json) {
        return $null
    }

    $record = $json | ConvertFrom-Json
    if (-not $record.PayloadHex) {
        return $null
    }

    $bytes = Convert-HexToBytes ([string]$record.PayloadHex)
    $decoded = Decode-BroadcastPayload $bytes
    if (-not $decoded) {
        return $null
    }

    $sessionId = if ($record.SessionId) { [string]$record.SessionId } elseif ($record.Value) { [string]$record.Value } else { $null }
    $timestampValue = if ($null -ne $record.TimestampUtc) { $record.TimestampUtc } elseif ($null -ne $record.WrittenAtUtc) { $record.WrittenAtUtc } else { $null }
    $timestamp = Convert-LoopbackTimestamp $timestampValue

    return [pscustomobject]@{
        Timestamp        = $timestamp
        TimestampUtc     = $timestamp
        BluetoothAddress = 'loopback'
        Rssi             = $null
        CompanyId        = [uint16]$record.CompanyId
        SessionId        = $sessionId
        ValueKind        = $decoded.ValueKind
        Value            = $decoded.Value
        PredictionScore  = if ($null -ne $record.PredictionScore) { [double]$record.PredictionScore } else { 1.0 }
        PayloadHex       = ([string]$record.PayloadHex)
    }
}

$script:FoundMatch = $false
$watcher = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher]::new()
$watcher.ScanningMode = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEScanningMode]::Active

$handler = [Windows.Foundation.TypedEventHandler[
    Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher,
    Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementReceivedEventArgs
]]{
    param($sender, $eventArgs)

    foreach ($manufacturerData in $eventArgs.Advertisement.ManufacturerData) {
        if ($manufacturerData.CompanyId -ne $CompanyId) {
            continue
        }

        $reader = [Windows.Storage.Streams.DataReader]::FromBuffer($manufacturerData.Data)
        $bytes = New-Object byte[] $manufacturerData.Data.Length
        $reader.ReadBytes($bytes)

        $decoded = Decode-BroadcastPayload $bytes
        if (-not $decoded) {
            continue
        }

        $script:FoundMatch = $true
        $timestampUtc = [DateTimeOffset]::UtcNow
        $record = [pscustomobject]@{
            Timestamp        = $timestampUtc
            TimestampUtc     = $timestampUtc
            BluetoothAddress = ('0x{0:X}' -f $eventArgs.BluetoothAddress)
            Rssi             = $eventArgs.RawSignalStrengthInDBm
            CompanyId        = $manufacturerData.CompanyId
            SessionId        = $decoded.Value
            ValueKind        = $decoded.ValueKind
            Value            = $decoded.Value
            PredictionScore  = 1.0
            PayloadHex       = ([System.BitConverter]::ToString($bytes) -replace '-', '')
        }

        if ($Raw) {
            $record | Format-List
        } else {
            Write-Output $record
        }

        if ($FirstOnly) {
            $sender.Stop()
        }
    }
}

[void]$watcher.add_Received($handler)
$watcher.Start()
Start-Sleep -Milliseconds 500

if ($watcher.Status.ToString() -ne "Started") {
    $status = $watcher.Status.ToString()
    $watcher.Stop()
    throw "BLE watcher did not start. Watcher status: $status"
}

try {
    Write-Host "Scanning for BLE values with company ID $CompanyId for up to $TimeoutSeconds seconds..."
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $loopbackPath = $null
    $loopbackLastWriteUtc = [DateTime]::MinValue

    if ($LoopbackJsonPath) {
        if ([System.IO.Path]::IsPathRooted($LoopbackJsonPath)) {
            $loopbackPath = [System.IO.Path]::GetFullPath($LoopbackJsonPath)
        } else {
            $loopbackPath = [System.IO.Path]::GetFullPath((Join-Path $PWD $LoopbackJsonPath))
        }
    }

    while ((Get-Date) -lt $deadline -and -not $script:FoundMatch) {
        if ($loopbackPath -and (Test-Path -LiteralPath $loopbackPath)) {
            $item = Get-Item -LiteralPath $loopbackPath
            if ($item.LastWriteTimeUtc -gt $loopbackLastWriteUtc) {
                $loopbackLastWriteUtc = $item.LastWriteTimeUtc
                $record = & $ReadLoopbackPayload $loopbackPath
                if ($record) {
                    $script:FoundMatch = $true
                    if ($Raw) {
                        $record | Format-List
                    } else {
                        Write-Output $record
                    }
                    if ($FirstOnly) {
                        break
                    }
                }
            }
        }

        Start-Sleep -Milliseconds 250
    }
} finally {
    $watcher.Stop()
}

if (-not $script:FoundMatch) {
    Write-Host "No matching BLE broadcasts were observed."
}
