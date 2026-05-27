@echo off
REM HQNNDL v3.0 — Launch Script (Windows)
echo.
echo ══════════════════════════════════════════════════
echo   HQNNDL v3.0  ^|  Quantum CSR Detection
echo ══════════════════════════════════════════════════

cd /d "%~dp0"

REM Find Python
where python >nul 2>&1 || (echo [ERROR] Python 3.9+ not found. Install from python.org && pause && exit /b 1)

REM Create venv if needed
if not exist ".venv\" (
    echo   Creating virtual environment...
    python -m venv .venv
)
call .venv\Scripts\activate.bat

REM Install dependencies
echo   Checking dependencies...
pip install -q -r requirements.txt

REM Create runtime dirs
if not exist "uploads" mkdir uploads
if not exist "results\reports" mkdir results\reports
if not exist "results\heatmaps" mkdir results\heatmaps

echo.
echo   Starting server...
echo   Open: http://localhost:5050
echo ══════════════════════════════════════════════════
echo.

cd backend
python app.py
pause
