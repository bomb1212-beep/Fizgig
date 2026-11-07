@echo off
cd /d "%~dp0"
if not exist "%~dp0venv\Scripts\pythonw.exe" (
    echo Fizgig Python is missing. Run install_fizgig.bat first.
    pause
    exit /b 1
)
start "" "%~dp0venv\Scripts\pythonw.exe" "%~dp0automation\dataset_gui.py" %*
