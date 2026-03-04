#!/bin/bash

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate edge

# Run transcriber
echo "[MedExtract] Starting autonomous transcription and extraction..."
python3 transcriber.py
