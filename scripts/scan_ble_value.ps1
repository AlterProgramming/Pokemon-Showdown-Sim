param(
    [uint16]$CompanyId = 65534,
    [int]$TimeoutSeconds = 30,
    [switch]$FirstOnly,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"

Add-Type -AssemblyName System.Runtime.WindowsRuntime
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEScanningMode, Windows, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.DataReader, Windows, ContentType = WindowsRuntime] | Out-Null

function Convert-BytesToUuidString {
    param([byte[]]$Bytes)

    if ($Bytes.Length -ne 16) {
        throw "UUID payloads must be exactly 16 bytes."
    }

    $hex = [System.BitConverter]::ToString($Bytes).Replace('-', '').ToLowerInvariant()
    return "{0}-{1}-{2}-{3}-{4}" -f $hex.Substring(0, 8), $hex.Substring(8, 4), $hex.Substring(12, 4), $hex.Substring(16, 4), $hex.Substring(20, 12)
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
        $record = [pscustomobject]@{
            Timestamp        = [DateTimeOffset]::Now
            BluetoothAddress = ('0x{0:X}' -f $eventArgs.BluetoothAddress)
            Rssi             = $eventArgs.RawSignalStrengthInDBm
            CompanyId        = $manufacturerData.CompanyId
            ValueKind        = $decoded.ValueKind
            Value            = $decoded.Value
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
    Start-Sleep -Seconds $TimeoutSeconds
} finally {
    $watcher.Stop()
}

if (-not $script:FoundMatch) {
    Write-Host "No matching BLE broadcasts were observed."
}
