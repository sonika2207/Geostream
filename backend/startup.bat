@echo off
REM Startup script for Windows
set PORT=%PORT:8000%
python -m uvicorn main:app --host 0.0.0.0 --port %PORT%
