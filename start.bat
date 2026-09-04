@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Jing Zhou

where py >nul 2>&1
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
where python >nul 2>&1
if errorlevel 1 (
  echo [径舟] 未找到 Python 3.10+
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [径舟] 创建 .venv
  %PY% -m venv .venv
)
call ".venv\Scripts\activate.bat"

echo [径舟] 安装依赖
python -m pip install -U pip -q
python -m pip install -r requirements.txt -q

if not exist ".env" (
  copy /Y ".env.example" ".env" >nul
  echo [径舟] 已生成 .env
  set /p KEY=[径舟] 粘贴 LLM_API_KEY: 
  if "%KEY%"=="" (
    echo [径舟] 未填写 Key
    pause
    exit /b 1
  )
  set /p URL=[径舟] API 地址 [回车=https://api.openai.com/v1]: 
  if "%URL%"=="" set URL=https://api.openai.com/v1
  set /p MODEL=[径舟] 模型名 [回车=gpt-4o]: 
  if "%MODEL%"=="" set MODEL=gpt-4o
  python -c "from pathlib import Path; p=Path('.env'); t=p.read_text(encoding='utf-8'); t=t.replace('LLM_API_KEY=sk-your-key-here','LLM_API_KEY='+r'''%KEY%'''); t=t.replace('LLM_BASE_URL=https://api.openai.com/v1','LLM_BASE_URL='+r'''%URL%'''); t=t.replace('LLM_MODEL=gpt-4o','LLM_MODEL='+r'''%MODEL%'''); p.write_text(t, encoding='utf-8')"
)

for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8777 ^| findstr LISTENING') do taskkill /f /pid %%a 2>nul
start "JingZhou Server" cmd /c "cd /d %~dp0 && call .venv\Scripts\activate.bat && python server.py"
:wait_loop
timeout /t 2 /nobreak >nul
netstat -ano 2>nul | findstr ":8777" | findstr "LISTENING" >nul
if errorlevel 1 goto wait_loop
start "" http://127.0.0.1:8777
echo [径舟] http://127.0.0.1:8777
pause
