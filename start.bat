@echo off
cd /d "%~dp0"
title Jing Zhou

echo.
echo   ========================================
echo         Jing Zhou
echo   ========================================
echo.

if not exist ".env" (
    echo [ERROR] .env not found
    pause
    exit /b 1
)

echo [1/3] Killing old server on port 8777...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8777 ^| findstr LISTENING') do (
    taskkill /f /pid %%a 2>nul
)

echo [2/3] Installing dependencies...
pip install -r requirements.txt -q 2>nul

echo [3/3] Starting server...
echo.
echo        Starting server, please wait...
start "JingZhou Server" cmd /c "cd /d %~dp0 && python server.py"

:wait_loop
timeout /t 2 /nobreak >nul
netstat -ano 2>nul | findstr ":8777" | findstr "LISTENING" >nul
if errorlevel 1 (
    echo        Still waiting...
    goto wait_loop
)

echo        Server ready!
start "" http://127.0.0.1:8777
echo.
echo        http://127.0.0.1:8777
echo        Close the server window to stop.
echo.
pause