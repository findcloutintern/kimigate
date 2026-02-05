@echo off
setlocal enabledelayedexpansion

echo.
echo  kimigate installer
echo  ================================================
echo.

set "KIMI_DIR=%USERPROFILE%\kimigate"
set "SHORTCUTS_DIR=%USERPROFILE%\shortcuts"
set "UV_PATH=%USERPROFILE%\.local\bin\uv.exe"
set "REPO_URL=https://github.com/findcloutintern/kimigate.git"

where git >nul 2>&1
if %errorlevel% neq 0 (
    echo error: git is not installed
    echo install git from https://git-scm.com/downloads
    pause
    exit /b 1
)

where claude >nul 2>&1
if %errorlevel% neq 0 (
    echo error: claude code is not installed
    echo install with: npm install -g @anthropic-ai/claude-code
    pause
    exit /b 1
)

if not exist "%UV_PATH%" (
    echo installing uv...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    if not exist "%UV_PATH%" (
        echo failed to install uv - please restart terminal and run again
        pause
        exit /b 1
    )
)

echo.
echo get your free api key at: https://build.nvidia.com
echo.
set /p "API_KEY=enter your nvidia nim api key: "
if "!API_KEY!"=="" (
    echo no api key provided
    pause
    exit /b 1
)

if exist "%KIMI_DIR%\.git" (
    echo updating existing installation...
    cd /d "%KIMI_DIR%"
    git pull
) else (
    if exist "%KIMI_DIR%" (
        echo removing old installation...
        rmdir /s /q "%KIMI_DIR%"
    )
    echo cloning kimigate...
    git clone "%REPO_URL%" "%KIMI_DIR%"
    if %errorlevel% neq 0 (
        echo failed to clone repo
        pause
        exit /b 1
    )
)

cd /d "%KIMI_DIR%"

echo creating config...
(
echo NVIDIA_NIM_API_KEY=!API_KEY!
echo RATE_LIMIT=40
echo RATE_WINDOW=60
echo TEMPERATURE=1.0
echo MAX_TOKENS=81920
echo SKIP_QUOTA_CHECK=true
echo SKIP_TITLE_GENERATION=true
echo SKIP_SUGGESTION_MODE=true
echo SKIP_FILEPATH_EXTRACTION=true
echo FAST_PREFIX_DETECTION=true
) > "%KIMI_DIR%\.env"

echo installing dependencies...
"%UV_PATH%" sync
if %errorlevel% neq 0 (
    echo failed to install dependencies
    pause
    exit /b 1
)

if not exist "%SHORTCUTS_DIR%" (
    echo creating shortcuts folder...
    mkdir "%SHORTCUTS_DIR%"
)

echo creating kimigate command...
(
echo @echo off
echo start /b "" cmd /c "cd /d %KIMI_DIR% ^&^& %UV_PATH% run python server.py ^>nul 2^>^&1"
echo timeout /t 3 /nobreak ^>nul
echo set ANTHROPIC_AUTH_TOKEN=kimigate
echo set ANTHROPIC_BASE_URL=http://localhost:8082
echo claude --dangerously-skip-permissions %%*
) > "%SHORTCUTS_DIR%\kimigate.bat"

echo %PATH% | find /i "%SHORTCUTS_DIR%" >nul
if %errorlevel% neq 0 (
    echo adding shortcuts to PATH...
    powershell -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path', 'User') + ';%SHORTCUTS_DIR%', 'User')"
)

echo.
echo  ================================================
echo  installation complete!
echo  ================================================
echo.
echo  open a NEW terminal and run: kimigate
echo.

endlocal
pause
