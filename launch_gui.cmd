@echo off
where pwsh.exe >nul 2>&1
if errorlevel 1 (
  powershell.exe -NoProfile -STA -File "%~dp0apps\predict_gui.ps1" %*
) else (
  pwsh.exe -NoProfile -STA -File "%~dp0apps\predict_gui.ps1" %*
)
if errorlevel 1 pause
