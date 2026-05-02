@echo off
echo ==========================================
echo   FX Arbitrage Detection Engine
echo   Starting FastAPI Backend + React Frontend
echo ==========================================
echo.

REM Start FastAPI backend in a new window
start "FX Arb Backend" cmd /k "cd /d "%~dp0backend" && "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak > nul

REM Start React frontend in another window
start "FX Arb Frontend" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ==========================================
echo   Backend:  http://localhost:8000
echo   Frontend: http://localhost:5173
echo   API Docs: http://localhost:8000/docs
echo ==========================================
echo.
echo Both servers starting in separate windows.
echo Press any key to close this launcher.
pause > nul
