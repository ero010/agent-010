@echo off
cd /d "%~dp0"
if not exist .env ( copy .env.example .env )
python -m uvicorn backend:app --port 8000
pause
