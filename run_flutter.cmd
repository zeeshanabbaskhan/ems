@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0app_frontend\run_flutter.ps1" %*
