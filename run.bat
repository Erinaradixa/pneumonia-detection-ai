@echo off
chcp 65001 > nul
title Pneumonia Detection App
echo ========================================
echo    PNEUMONIA DETECTION AI APP
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version > nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo SUCCESS: Python detected
python --version

echo.
echo [2/4] Installing required packages...
pip install -r requirements.txt

if errorlevel 1 (
    echo WARNING: Some packages failed to install
    echo Trying to continue anyway...
    timeout /t 3 /nobreak > nul
)

echo.
echo [3/4] Starting Pneumonia Detection App...
echo ========================================
echo    OPEN YOUR BROWSER AND GO TO:
echo        http://localhost:5000
echo ========================================
echo.
echo FEATURES:
echo - Upload X-ray images
echo - AI Pneumonia Detection  
echo - Heatmap Visualization
echo - Real-time Analysis
echo.
echo Press CTRL+C to stop the application
echo.

python app.py
pause