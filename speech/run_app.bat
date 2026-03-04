@echo off
echo.
echo [MedExtract] Activating conda environment 'edge'...
call conda activate edge

echo [MedExtract] Starting autonomous transcription and extraction...
python transcriber.py

pause
