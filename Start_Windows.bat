@echo off
REM Windows launcher for Puzzle Hand Analyzer.
REM Uses a local virtualenv (.venv) to avoid messing with the system Python.

cd /d "%~dp0"

echo ============================================
echo    Puzzle Hand Analyzer
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed, or it is not on your PATH.
    echo.
    echo Install Python 3.10 or newer from:
    echo     https://www.python.org/downloads/
    echo.
    echo IMPORTANT: on the first installer screen, tick the box
    echo            "Add Python to PATH"  before clicking Install.
    echo.
    pause
    exit /b 1
)

if not exist .venv (
    echo Creating local environment ^(one-time, about 30 seconds^)...
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo Could not create the local environment.
        echo Make sure Python is installed correctly and try again.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

echo Checking dependencies (the first run takes a few minutes)...
python -m pip install --quiet --upgrade pip
REM Don't quiet this one - mediapipe is ~100 MB; without progress output the
REM install looks like a hang and people force-close the launcher.
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo Dependency installation failed. See the messages above.
    pause
    exit /b 1
)

echo Launching...
python puzzle_app.py
