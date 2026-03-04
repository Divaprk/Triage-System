@echo off
echo.
echo [MedExtract] Creating conda environment 'edge'...
call conda create -n edge python=3.10 -y

echo [MedExtract] Activating environment...
call conda activate edge

echo [MedExtract] Upgrading pip...
python -m pip install --upgrade pip

echo [MedExtract] Installing requirements. This may take a few minutes...
pip install -r requirements.txt

echo.
echo [COMPLETE] Setup finished. Use run_app.bat to start.
pause
