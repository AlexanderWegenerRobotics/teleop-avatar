@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

rem --- optional role override, forwarded as-is to both avatar.exe and
rem     avatar_pipeline.exe (see twin/role.hpp parseRoleFlag). Omit to use
rem     whatever config.yaml's role: key says -- unchanged from before this
rem     flag existed. Usage: launch.bat --twin   |   launch.bat --avatar
set "ROLE_ARG=%~1"
if not "%ROLE_ARG%"=="" (
    if not "%ROLE_ARG%"=="--twin" if not "%ROLE_ARG%"=="--avatar" (
        echo [ERROR]: unrecognized argument "%ROLE_ARG%" ^(expected --twin or --avatar^)
        exit /b 1
    )
)

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

set "WORK_DIR=%SCRIPT_DIR%\build"
set "LOG_OUT=%SCRIPT_DIR%\avatar_stdout.log"
set "LOG_ERR=%SCRIPT_DIR%\avatar_stderr.log"
set "CRASH_LOG=%SCRIPT_DIR%\avatar_crash.log"
set "PS_FILE=%TEMP%\avatar_monitor_%RANDOM%.ps1"

if "%ROLE_ARG%"=="" (
    echo [LAUNCH]: Starting avatar  ^(role: config.yaml default^)  ^(stdout: avatar_stdout.log  stderr: avatar_stderr.log^)
) else (
    echo [LAUNCH]: Starting avatar  ^(role: %ROLE_ARG%^)  ^(stdout: avatar_stdout.log  stderr: avatar_stderr.log^)
)

rem --- Write the PowerShell monitor script line by line.
rem     Using individual >> redirections avoids cmd mis-parsing parentheses inside
rem     the (echo ... ) > file block form.  Parentheses in PS code that fall
rem     outside double-quoted strings are written as ^( ^) so cmd emits ( ).
del "%PS_FILE%" 2>nul

>>"%PS_FILE%" echo $avatarExe   = '%AVATAR%'
>>"%PS_FILE%" echo $streamerExe = '%STREAMER%'
>>"%PS_FILE%" echo $workDir     = '%WORK_DIR%'
>>"%PS_FILE%" echo $logOut      = '%LOG_OUT%'
>>"%PS_FILE%" echo $logErr      = '%LOG_ERR%'
>>"%PS_FILE%" echo $crashLog    = '%CRASH_LOG%'
>>"%PS_FILE%" echo $roleArg     = '%ROLE_ARG%'
>>"%PS_FILE%" echo.
>>"%PS_FILE%" echo $avatarParams = @{
>>"%PS_FILE%" echo     FilePath               = $avatarExe
>>"%PS_FILE%" echo     WorkingDirectory       = $workDir
>>"%PS_FILE%" echo     RedirectStandardOutput = $logOut
>>"%PS_FILE%" echo     RedirectStandardError  = $logErr
>>"%PS_FILE%" echo     PassThru               = $true
>>"%PS_FILE%" echo     NoNewWindow            = $true
>>"%PS_FILE%" echo }
>>"%PS_FILE%" echo if ^($roleArg -ne ''^) { $avatarParams['ArgumentList'] = $roleArg }
>>"%PS_FILE%" echo $avProc = Start-Process @avatarParams
>>"%PS_FILE%" echo if ^(-not $avProc^) { Write-Host '[ERROR]: Failed to start avatar.exe'; exit 1 }
>>"%PS_FILE%" echo $avPid = $avProc.Id
>>"%PS_FILE%" echo Write-Host "[LAUNCH]: avatar PID=$avPid"
>>"%PS_FILE%" echo.
>>"%PS_FILE%" echo Start-Sleep -Seconds 2
>>"%PS_FILE%" echo.
>>"%PS_FILE%" echo $streamerParams = @{
>>"%PS_FILE%" echo     FilePath         = $streamerExe
>>"%PS_FILE%" echo     WorkingDirectory = $workDir
>>"%PS_FILE%" echo     PassThru         = $true
>>"%PS_FILE%" echo     NoNewWindow      = $true
>>"%PS_FILE%" echo }
>>"%PS_FILE%" echo if ^($roleArg -ne ''^) { $streamerParams['ArgumentList'] = $roleArg }
>>"%PS_FILE%" echo $stProc = Start-Process @streamerParams
>>"%PS_FILE%" echo if ^(-not $stProc^) {
>>"%PS_FILE%" echo     Write-Host '[ERROR]: Failed to start avatar_pipeline.exe'
>>"%PS_FILE%" echo     Stop-Process -Id $avProc.Id -Force -EA SilentlyContinue
>>"%PS_FILE%" echo     exit 1
>>"%PS_FILE%" echo }
>>"%PS_FILE%" echo $stPid = $stProc.Id
>>"%PS_FILE%" echo Write-Host "[LAUNCH]: streamer PID=$stPid"
>>"%PS_FILE%" echo Write-Host '[LAUNCH]: Press Ctrl+C to stop both.'
>>"%PS_FILE%" echo.
>>"%PS_FILE%" echo function Show-Log {
>>"%PS_FILE%" echo     param^([string]$Path, [string]$Label, [int]$MaxLines = 30^)
>>"%PS_FILE%" echo     if ^(Test-Path $Path^) {
>>"%PS_FILE%" echo         $lines = Get-Content $Path -Tail $MaxLines
>>"%PS_FILE%" echo         if ^($lines^) {
>>"%PS_FILE%" echo             Write-Host "--- $Label ---"
>>"%PS_FILE%" echo             $lines ^| ForEach-Object { Write-Host "  $_" }
>>"%PS_FILE%" echo             Write-Host '---'
>>"%PS_FILE%" echo         }
>>"%PS_FILE%" echo     }
>>"%PS_FILE%" echo }
>>"%PS_FILE%" echo.
>>"%PS_FILE%" echo try {
>>"%PS_FILE%" echo     while ^($true^) {
>>"%PS_FILE%" echo         if ^($avProc.HasExited^) {
>>"%PS_FILE%" echo             $code = $avProc.ExitCode
>>"%PS_FILE%" echo             Write-Host ''
>>"%PS_FILE%" echo             if ^($code -eq 0^) {
>>"%PS_FILE%" echo                 Write-Host '[LAUNCH]: avatar exited cleanly.'
>>"%PS_FILE%" echo             } else {
>>"%PS_FILE%" echo                 Write-Host "[LAUNCH]: avatar exited with error code $code."
>>"%PS_FILE%" echo             }
>>"%PS_FILE%" echo             Show-Log $logOut 'avatar stdout'
>>"%PS_FILE%" echo             Show-Log $logErr 'avatar stderr'
>>"%PS_FILE%" echo             if ^(Test-Path $crashLog^) { Show-Log $crashLog 'crash log' 100 }
>>"%PS_FILE%" echo             break
>>"%PS_FILE%" echo         }
>>"%PS_FILE%" echo         if ^($stProc.HasExited^) {
>>"%PS_FILE%" echo             $stCode = $stProc.ExitCode
>>"%PS_FILE%" echo             Write-Host "[LAUNCH]: avatar_pipeline exited with code $stCode."
>>"%PS_FILE%" echo             break
>>"%PS_FILE%" echo         }
>>"%PS_FILE%" echo         Start-Sleep -Milliseconds 500
>>"%PS_FILE%" echo     }
>>"%PS_FILE%" echo } finally {
>>"%PS_FILE%" echo     Write-Host ''
>>"%PS_FILE%" echo     Write-Host '[LAUNCH]: Shutting down...'
>>"%PS_FILE%" echo     if ^(-not $stProc.HasExited^) { Stop-Process -Id $stProc.Id -Force -EA SilentlyContinue }
>>"%PS_FILE%" echo     if ^(-not $avProc.HasExited^) { Stop-Process -Id $avProc.Id -Force -EA SilentlyContinue }
>>"%PS_FILE%" echo     Write-Host '[LAUNCH]: All processes stopped.'
>>"%PS_FILE%" echo }

powershell -NoProfile -ExecutionPolicy Bypass -File "%PS_FILE%"
del "%PS_FILE%" 2>nul
