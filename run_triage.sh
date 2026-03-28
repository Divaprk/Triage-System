#!/bin/bash
echo "========================================"
echo "  Pi 5 Triage System — Starting up"
echo "========================================"

cd ~/triage_system
source venv/bin/activate
echo "[run_triage] Virtual environment activated."
echo "[run_triage] Launching orchestrator..."
echo ""
python3.11 core/orchestrator2.py
