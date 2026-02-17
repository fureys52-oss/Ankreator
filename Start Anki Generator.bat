@echo off
title Ankreator

:: ===================================================================
::  This script launches the Anki Deck Generator application
::  using the dedicated virtual environment.
:: ===================================================================

cd /d "%~dp0"

:: Check if the virtual environment exists. If not, tell the user to run setup.
IF NOT EXIST ".\venv\Scripts\python.exe" (
    echo.
    echo ERROR: The virtual environment was not found.
    echo Please run the "setup.bat" script once to create it.
    echo.
    pause
    exit
)

echo Launching Ankreator

:: Run the application using the Python from our virtual environment.
:: This automatically handles all package paths correctly.
call ".\venv\Scripts\python.exe" app.py

echo.
echo Application finished. Press any key to close this window.
pause >nul