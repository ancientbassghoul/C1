@echo off

echo Checking Python version...
python -c "import sys; code = 0 if sys.version_info[:2] == (3,11) else 1; print(f'Found Python {sys.version_info.major}.{sys.version_info.minor}'); sys.exit(code)"
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Python 3.11 is required.
    echo Please install it from: https://www.python.org/downloads/release/python-3119/
    echo Then re-run this script.
    pause
    exit /b 1
)

echo Setting up virtual environment...
python -m venv venv

echo Installing PyTorch (CPU)...
venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo Installing dependencies...
venv\Scripts\pip install -r requirements.txt

echo Installing local package (registers pipeline module)...
venv\Scripts\pip install -e .

echo.
echo Done. To run:
echo   venv\Scripts\python raycast.py --frames_dir ./frames
