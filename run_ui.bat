@echo off
title CohereX Studio Launcher
echo ===================================================
echo           Starting CohereX Studio Web UI
echo ===================================================
echo Activating Conda environment 'coherex'...
call "%USERPROFILE%\miniconda3\Scripts\activate.bat" coherex
if %ERRORLEVEL% NEQ 0 (
    echo Error activating conda environment. Trying direct python...
)

echo.
echo Launching Streamlit interface on http://localhost:8501 (Max upload size: 50 GB) ...
"%USERPROFILE%\miniconda3\envs\coherex\python.exe" -m streamlit run "%~dp0app.py" --server.port 8501 --server.maxUploadSize 50000 --server.headless false

pause
