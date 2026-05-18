@echo off

echo Checking venv exists...
if not exist venv\Scripts\pip.exe (
    echo ERROR: venv not found. Run SETUP_FIRST.bat first.
    pause
    exit /b 1
)

echo Installing pyceres (Ceres Solver orientation optimizer)...
venv\Scripts\pip install pyceres

echo Installing GeoCalib (single-image pitch estimation)...
venv\Scripts\pip install geocalib

echo.
echo ═══════════════════════════════════════════════════════════════
echo   SETUP_SECOND complete.  All dependencies installed.
echo.
echo   To run:
echo     venv\Scripts\python raycast.py --frames_dir ./frames
echo ═══════════════════════════════════════════════════════════════
