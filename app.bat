@echo off
setlocal

cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" goto :run

echo Creating virtual environment in .venv ...
python -m venv .venv
if errorlevel 1 (
    echo Failed to create virtual environment. Is Python installed and on PATH?
    exit /b 1
)

echo Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed.
    exit /b 1
)

echo Applying Django migrations...
".venv\Scripts\python.exe" webui\manage.py migrate

:run
echo Starting server on 0.0.0.0:8000 (reachable at http://100.121.87.117:8000 and http://localhost:8000) ...
".venv\Scripts\python.exe" webui\manage.py runserver 0.0.0.0:8000

endlocal
