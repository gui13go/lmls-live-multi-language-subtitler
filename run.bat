@echo off
REM Quick launcher script for Windows
cd /d "%~dp0"
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)
python main.py %*
