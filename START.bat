@echo off
cd /d "%~dp0"
if not exist .env ( copy .env.example .env )
set LOCAL_PC=1
python -m uvicorn backend:app --port 8000
pause
