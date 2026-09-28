@echo off

REM Go to project root
cd /d %~dp0..

echo Starting backend API...
start "Backend API" cmd /k "call .venv\Scripts\activate && uvicorn app.api:app"

echo Starting frontend...
start "Frontend UI" cmd /k "call .venv\Scripts\activate && streamlit run app/frontend.py"

echo Waiting for services to start...
timeout /t 5 > nul

echo Opening browser...
start http://127.0.0.1:8501