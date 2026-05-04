# Use Flutter from android/local.properties and avoid space-in-path bugs (Zeeshan Abbas\...)
# by preferring a space-free PUB_CACHE. Run from app_frontend:
#   .\run_flutter.cmd build apk --release   (works if .ps1 is blocked by execution policy)
#   .\run_flutter.ps1 build apk --release

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $FlutterArgs
)

$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$propsPath = Join-Path $here "android\local.properties"
if (-not (Test-Path $propsPath)) {
    throw "Missing $propsPath"
}

$flutterSdk = $null
Get-Content -LiteralPath $propsPath | ForEach-Object {
    if ($_ -match '^\s*flutter\.sdk\s*=\s*(.+)\s*$') {
        $flutterSdk = $Matches[1].Trim()
    }
}
if (-not $flutterSdk) {
    throw "flutter.sdk not found in android\local.properties"
}
# Properties file uses doubled backslashes
$flutterSdk = $flutterSdk.Replace('\\', '\')
$flutterBat = Join-Path $flutterSdk "bin\flutter.bat"
if (-not (Test-Path $flutterBat)) {
    throw "Flutter not found at $flutterBat - fix flutter.sdk in android\local.properties"
}

if (-not $env:PUB_CACHE) {
    $env:PUB_CACHE = "D:\pub_cache"
}
New-Item -ItemType Directory -Force -Path $env:PUB_CACHE | Out-Null

$env:FLUTTER_ROOT = $flutterSdk
$bin = Join-Path $flutterSdk "bin"
$env:PATH = "$bin;$env:PATH"

Set-Location -LiteralPath $here
& $flutterBat @FlutterArgs
exit $LASTEXITCODE
