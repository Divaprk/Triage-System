#!/bin/bash
set -e

echo "[MedExtract] Creating conda environment 'edge' with Python 3.10..."
conda create -n edge python=3.10 -y

echo "[MedExtract] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate edge

echo "[MedExtract] Upgrading pip..."
python -m pip install --upgrade pip

echo "[MedExtract] Installing requirements..."
pip install -r requirements.txt

echo -e "\n[COMPLETE] Setup finished. Use ./run_app.sh to start."
