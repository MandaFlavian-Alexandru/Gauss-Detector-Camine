@echo off
title Gauss Detector Camine - Launcher
echo =========================================
echo Starting Gauss Detector Camine
echo =========================================

cd /d "%~dp0"

:: Check and setup Python environment
IF NOT EXIST ".venv" (
    echo [1/4] Creating Python Virtual Environment...
    python -m venv .venv
    echo [2/4] Installing Python dependencies...
    call .venv\Scripts\activate.bat
    pip install -r requirements.txt
    deactivate
) ELSE (
    echo [1/2] Python environment found.
)

:: Check and setup Node.js environment
IF NOT EXIST "frontend\node_modules" (
    echo [3/4] Installing Node.js dependencies...
    cd frontend
    call npm install
    cd ..
) ELSE (
    echo [2/2] Node.js environment found.
)

:: Build Next.js if missing
IF NOT EXIST "frontend\.next" (
    echo Building Next.js Production Build...
    cd frontend
    call npm run build
    cd ..
)

echo.
echo Starting Services...
echo.

:: Start FastAPI Backend in a new window
start "Gauss Backend" cmd /k "call .venv\Scripts\activate.bat && set PYTHONPATH=%cd% && uvicorn Gauss_MD_API:app --host 0.0.0.0 --port 8001 --app-dir ."

:: Start Next.js Frontend in a new window
start "Gauss Frontend" cmd /k "cd frontend && npm start -- -p 3001"

echo Application started! 
echo - Dashboard: http://localhost:3001
echo - Backend:   http://localhost:8001
echo.
echo You can close this launcher window. The services are running in the newly opened windows.
pause
