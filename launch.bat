@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem --- locate binaries (Release preferred, Debug fallback) ---
set "AVATAR="
set "STREAMER="
for %%C in (Release Debug) do (
    if not defined AVATAR (
        if exist "%SCRIPT_DIR%\build\%%C\avatar.exe" (
            set "AVATAR=%SCRIPT_DIR%\build\%%C\avatar.exe"
            set "STREAMER=%SCRIPT_DIR%\build\%%C\avatar_pipeline.exe"
        )
    )
)

rem --- add MuJoCo DLL to PATH if not already there ---
if exist "C:\dev\mujoco-3.3.0\bin\mujoco.dll" (
    set "PATH=C:\dev\mujoco-3.3.0\bin;%PATH%"
)

rem --- sanity checks ---
if not defined AVATAR (
    echo [ERROR]: avatar.exe not found in build\Release or build\Debug
    exit /b 1
)
if not exist "%STREAMER%" (
    echo [ERROR]: avatar_pipeline.exe not found at %STREAMER%
    exit /b 1
)

rem --- working directory for both processes: must be build/ so ../config and ../models resolve ---
set "WORK_DIR=%SCRIPT_DIR%\build"

echo [LAUNCH]: Starting avatar...
start /D "%WORK_DIR%" "" /B "%AVATAR%"

rem Capture PID of the most recently started avatar.exe via PowerShell
for /f %%P in ('powershell -NoProfile -Command ^
    "Get-Process avatar -EA SilentlyContinue | Sort-Object StartTime -Desc | Select -First 1 -Exp Id"') do set "AVATAR_PID=%%P"

if not defined AVATAR_PID (
    echo [ERROR]: Failed to get avatar PID.
    exit /b 1
)

timeout /t 2 /nobreak > nul

echo [LAUNCH]: Starting avatar_pipeline...
start /D "%WORK_DIR%" "" /B "%STREAMER%"

for /f %%P in ('powershell -NoProfile -Command ^
    "Get-Process avatar_pipeline -EA SilentlyContinue | Sort-Object StartTime -Desc | Select -First 1 -Exp Id"') do set "STREAMER_PID=%%P"

if not defined STREAMER_PID (
    echo [ERROR]: Failed to get avatar_pipeline PID.
    if defined AVATAR_PID taskkill /PID %AVATAR_PID% /F >nul 2>&1
    exit /b 1
)

echo [LAUNCH]: avatar PID=%AVATAR_PID% ^| streamer PID=%STREAMER_PID%

rem --- hand off monitoring to PowerShell (try/finally survives Ctrl+C) ---
(
    echo $avId = %AVATAR_PID%
    echo $stId = %STREAMER_PID%
    echo $av = Get-Process -Id $avId -EA SilentlyContinue
    echo $st = Get-Process -Id $stId -EA SilentlyContinue
    echo Write-Host '[LAUNCH]: Press Ctrl+C to stop both.'
    echo try {
    echo     while ^($true^) {
    echo         if ^(-not $av -or $av.HasExited^) { Write-Host '[LAUNCH]: avatar exited.'; break }
    echo         if ^(-not $st -or $st.HasExited^) { Write-Host '[LAUNCH]: avatar_pipeline exited.'; break }
    echo         Start-Sleep -Milliseconds 500
    echo     }
    echo } finally {
    echo     Write-Host ''
    echo     Write-Host '[LAUNCH]: Shutting down...'
    echo     if ^($st -and -not $st.HasExited^) { Stop-Process -Id $stId -Force -EA SilentlyContinue }
    echo     if ^($av -and -not $av.HasExited^) { Stop-Process -Id $avId -Force -EA SilentlyContinue }
    echo     Write-Host '[LAUNCH]: All processes stopped.'
    echo }
) > "%TEMP%\avatar_monitor.ps1"
powershell -NoProfile -ExecutionPolicy Bypass -File "%TEMP%\avatar_monitor.ps1"
del "%TEMP%\avatar_monitor.ps1" > nul 2>&1
