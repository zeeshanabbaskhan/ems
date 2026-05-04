@echo off
REM Same as run_flutter.ps1; use when PowerShell blocks scripts (execution policy).
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_flutter.ps1" %*
