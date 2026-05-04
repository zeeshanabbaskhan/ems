param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $FlutterArgs
)
$inner = Join-Path $PSScriptRoot "app_frontend\run_flutter.ps1"
if (-not (Test-Path $inner)) {
    throw "Missing $inner"
}
& $inner @FlutterArgs
exit $LASTEXITCODE
