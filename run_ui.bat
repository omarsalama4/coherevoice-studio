@echo off
title CohereX Studio Launcher
echo ===================================================
echo           Starting CohereX Studio Web UI
echo ===================================================
echo Activating Conda environment 'coherex'...
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" coherex
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Could not activate the 'coherex' environment.
    echo Create it with: conda env create -f "%~dp0environment.yml"
    pause
    exit /b 1
)

echo.
echo Validating Python dependencies...
python "%~dp0scripts\doctor.py"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Environment validation failed. Follow the repair command above.
    pause
    exit /b 1
)

echo Launching Streamlit interface on http://127.0.0.1:8501 ...
python -m streamlit run "%~dp0app.py" --server.address 127.0.0.1 --server.port 8501 --server.maxUploadSize 2048 --server.headless false

pause
