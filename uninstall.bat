@echo off
setlocal

echo.
echo  kimigate uninstaller
echo  ================================================
echo.

set "KIMI_DIR=%USERPROFILE%\kimigate"
set "SHORTCUTS_DIR=%USERPROFILE%\shortcuts"
set "SHORTCUT=%SHORTCUTS_DIR%\kimigate.bat"

tasklist /fi "imagename eq python.exe" 2>nul | find /i "python" >nul
if %errorlevel% equ 0 (
    echo stopping server...
    taskkill /f /im python.exe >nul 2>&1
)

if exist "%SHORTCUT%" (
    echo removing shortcut...
    del "%SHORTCUT%"
)

if exist "%KIMI_DIR%" (
    echo removing kimigate folder...
    rmdir /s /q "%KIMI_DIR%"
)

echo.
echo  ================================================
echo  uninstall complete
echo  ================================================
echo.

endlocal
pause
