# Pi 5 Triage System

An edge-based patient triage system running on a Raspberry Pi 5. It fuses vision, vitals, and speech inputs to classify patients in real time as **CRITICAL**, **URGENT**, or **STABLE**.

---

## How It Works

Four threads run concurrently on the Pi, writing to a shared blackboard:

| Thread | Input | Output |
|---|---|---|
| Vision | Webcam | Eye Aspect Ratio (EAR) — detects alertness |
| Vitals | MAX30100 + MLX90614 | Heart rate, SpO2, temperature |
| Speech | Microphone (push-to-talk) | Chest pain, breathlessness flags |
| Inference | Blackboard | Triage level → HTTP POST to laptop |

A lightweight neural network (`triage_model.pth`) runs every 0.5s and classifies the patient based on all sensor inputs combined.

---

## Hardware Required

- Raspberry Pi 5
- MAX30100 pulse oximeter (I2C)
- MLX90614 IR temperature sensor (I2C)
- USB webcam
- USB microphone
- Tactile push button on GPIO17

---

## Folder Structure

```
triage_system/
├── README.md
├── .gitignore
├── laptop_receiver.py          # Run this on your laptop to view the dashboard
├── run_triage.sh               # Entry point for the Pi
├── requirements.txt            # Pi Python dependencies
├── triage_model.pth            # Trained triage neural network
│
├── core/                       # Runtime code
│   ├── orchestrator2.py        # Main thread manager + blackboard
│   ├── vision.py               # EAR calculation via MediaPipe
│   ├── pi_push_vitals.py       # HR / SpO2 / temp sensing
│   ├── max30100.py             # MAX30100 I2C driver
│   ├── transcriber_integrated2.py  # Push-to-talk VAD + ASR pipeline
│   ├── asr.py                  # Faster-Whisper wrapper
│   └── symptom_embedder.py     # KNN symptom detection
│
├── anchors/                    # Precomputed sentence embeddings
│   ├── chest_pain_pos_anchors.npy
│   ├── chest_pain_neg_anchors.npy
│   ├── breathlessness_pos_anchors.npy
│   └── breathlessness_neg_anchors.npy
│
└── benchmarks/                 # Profiling scripts (not part of runtime)
    ├── diagnostics.py
    └── paso_benchmarks.py
```

---

## Setup

**1. Clone the repo onto your Pi:**
```bash
git clone <repo-url> ~/triage_system
cd ~/triage_system
```

**2. Create and activate a virtual environment:**
```bash
python3.11 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

> This will take a while on the first run due to PyTorch and the sentence-transformers model.

---

## Running the System

```bash
bash run_triage.sh
```

This activates the venv and launches the orchestrator. All four threads start automatically.

**Push-to-Talk:** Hold the button on GPIO17 to speak. Release when done — the system will transcribe and detect symptoms automatically.

**Shutdown:** Press `Ctrl+C` for a clean shutdown.

---

## Receiving Data (Laptop Side)

Run `laptop_receiver.py` on your laptop to see the live dashboard:

```bash
pip install flask flask-socketio
python laptop_receiver.py
```

Then open your browser at `http://127.0.0.1:5001`.

The Pi POSTs triage results every 0.5s to:
```
http://192.168.137.1:5001/update_vitals
```

Make sure your laptop is connected to the Pi via USB network sharing (ICS) with the laptop IP set to `192.168.137.1`.

---

## Dependencies

Key packages (see `requirements.txt` for full list):

| Package | Purpose |
|---|---|
| `torch` | Triage neural network inference |
| `faster-whisper` | Speech-to-text (INT8, optimised for Pi) |
| `sentence-transformers` | Symptom embedding model |
| `mediapipe` | Face mesh / EAR calculation |
| `smbus2` | I2C communication with sensors |
| `gpiozero` | GPIO button control |
| `sounddevice` | Microphone audio capture |