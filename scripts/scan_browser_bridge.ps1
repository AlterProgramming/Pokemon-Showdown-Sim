param(
    [string]$BridgeUrl = "http://127.0.0.1:5000/bridge/messages",
    [int]$TimeoutSeconds = 30,
    [int]$AfterId = 0,
    [int]$Limit = 100,
    [switch]$FirstOnly,
    [switch]$Raw
)

$ErrorActionPreference = "Stop"

function Convert-BridgeTimestamp {
    param([object]$Value)

    if ($null -eq $Value) {
        return [DateTimeOffset]::UtcNow
    }

    if ($Value -is [DateTimeOffset]) {
        return $Value
    }

    if ($Value -is [DateTime] ) {
        return [DateTimeOffset]::new($Value.ToUniversalTime())
    }

    try {
        return [DateTimeOffset]::Parse([string]$Value)
    } catch {
        return [DateTimeOffset]::UtcNow
    }
}

function Invoke-BridgePoll {
    param(
        [string]$Url,
        [int]$Cursor,
        [int]$BatchLimit
    )

    $requestUrl = "$Url?after_id=$Cursor&limit=$BatchLimit"
    return Invoke-RestMethod -Method Get -Uri $requestUrl
}

Write-Host "Scanning browser bridge at $BridgeUrl for up to $TimeoutSeconds seconds..."
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$cursor = [math]::Max($AfterId, 0)
$found = $false

while ((Get-Date) -lt $deadline -and -not $found) {
    try {
        $response = Invoke-BridgePoll -Url $BridgeUrl -Cursor $cursor -BatchLimit $Limit
    } catch {
        throw "Failed to reach browser bridge at $BridgeUrl. Start the bridge server first. $($_.Exception.Message)"
    }

    foreach ($message in @($response.messages)) {
        if ($null -eq $message) {
            continue
        }

        $cursor = [int]$message.id
        $record = [pscustomobject]@{
            TimestampUtc    = Convert-BridgeTimestamp $message.timestamp_utc
            Id              = $message.id
            Source          = $message.source
            Target          = $message.target
            Kind            = $message.kind
            SessionId       = $message.session_id
            ValueKind       = $message.value_kind
            Value           = $message.value
            PredictionScore = if ($null -ne $message.prediction_score) { [double]$message.prediction_score } else { 1.0 }
            PayloadHex      = $message.payload_hex
            Note            = $message.note
        }

        $found = $true
        if ($Raw) {
            $record | Format-List
        } else {
            Write-Output $record
        }

        if ($FirstOnly) {
            break
        }
    }

    if (-not $found) {
        Start-Sleep -Milliseconds 250
    }
}

if (-not $found) {
    Write-Host "No browser bridge messages were observed."
}
