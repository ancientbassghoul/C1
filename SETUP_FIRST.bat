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

echo Installing core dependencies (includes transformers==4.38.1 required by GroundingDINO)...
venv\Scripts\pip install -r requirements.txt

echo Installing LightGlue (SuperPoint feature matcher)...
venv\Scripts\pip install git+https://github.com/cvg/LightGlue.git

echo Installing local package (registers pipeline module)...
venv\Scripts\pip install -e .

echo.
echo ═══════════════════════════════════════════════════════════════
echo   SETUP_FIRST complete.
echo.
echo   Next: install GroundingDINO manually.
echo   See README.md Setup section for the exact commands
echo   (requires VS Build Tools + CUDA + vcvars64.bat).
echo.
echo   Once GroundingDINO is installed, run SETUP_SECOND.bat.
echo ═══════════════════════════════════════════════════════════════
