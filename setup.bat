@echo off
title Ankreator v5.0 - Auto Setup

:: ===================================================================
::  Ankreator v5.0 Setup Script
::  1. Auto-installs Python 3.11 if missing.
::  2. Creates Virtual Environment & Installs Requirements.
::  3. Creates a Desktop Shortcut with your Icon.
:: ===================================================================

cd /d "%~dp0"
echo.
echo [1/4] Checking System Requirements...

:: --- CHECK FOR PYTHON ---
python --version >nul 2>&1
if %errorlevel% equ 0 goto :FOUND_PYTHON

echo.
echo [!] Python is not installed. 
echo     Downloading Python 3.11 automatically...
echo.

:: 1. Download Python 3.11 Installer (Silent)
curl -o python_installer.exe https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe

if not exist python_installer.exe (
    echo [ERROR] Download failed. Please install Python manually from python.org.
    pause
    exit /b 1
)

echo.
echo [!] Installing Python 3.11...
echo     (A window may pop up asking for permission - please click Yes)
echo.

:: 2. Install Python (Silent Mode)
python_installer.exe /quiet PrependPath=1 Include_test=0 Include_tcltk=0 InstallAllUsers=0

:: 3. Cleanup
del python_installer.exe

:: 4. Force Path Refresh
echo.
echo [!] Refreshing Environment...
set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts"

:: 5. Verification
python --version >nul 2>&1
if %errorlevel% neq 0 (
    if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
        goto :FOUND_PYTHON_MANUAL
    )
    echo [ERROR] Could not find Python after install. 
    echo Please close this window and run 'setup.bat' again.
    pause
    exit
)

:FOUND_PYTHON
set "PYTHON_CMD=python"

:FOUND_PYTHON_MANUAL
echo     Python found! Proceeding...

:: --- VIRTUAL ENV ---
echo.
echo [2/4] Checking Virtual Environment...
if exist ".\venv\Scripts\python.exe" (
    echo     Virtual Environment already exists. Skipping creation.
) else (
    echo     Creating venv...
    "%PYTHON_CMD%" -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create venv.
        pause
        exit /b 1
    )
)

:: --- INSTALL REQUIREMENTS ---
echo.
echo [3/4] Installing Dependencies...
echo     (This may take a few minutes for AI libraries...)
echo.

".\venv\Scripts\python.exe" -m pip install --upgrade pip
".\venv\Scripts\python.exe" -m pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Installation failed. Check the error message above.
    pause
    exit /b 1
)

:: --- CREATE SHORTCUT ---
echo.
echo [4/4] Creating Desktop Shortcut...

set "TARGET_SCRIPT=%~dp0Start Anki Generator.bat"
set "ICON_FILE=%~dp0icon.ico"
set "WORK_DIR=%~dp0"

:: 1. Dynamic Desktop Search (Fixes OneDrive Issues)
::    This asks Windows where the "Real" Desktop is.
for /f "usebackq tokens=*" %%D in (`powershell -command "[Environment]::GetFolderPath('Desktop')"` ) do set "REAL_DESKTOP=%%D"

set "SHORTCUT_PATH=%REAL_DESKTOP%\Ankreator.lnk"

:: 2. Create the Shortcut using the discovered path
powershell -Command "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%SHORTCUT_PATH%'); $s.TargetPath = '%TARGET_SCRIPT%'; $s.IconLocation = '%ICON_FILE%'; $s.WorkingDirectory = '%WORK_DIR%'; $s.Save()"

if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Could not create shortcut automatically.
    echo Please right-click 'Start Anki Generator.bat' and select 'More options -> Send to -> Desktop (create shortcut)'.
) else (
    echo.
    echo ===================================================
    echo    SETUP COMPLETE! 
    echo    1. You can close this window.
    echo    2. Look for the 'Ankreator' shortcut on your Desktop.
    echo ===================================================
)
echo.
pause