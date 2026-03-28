"""
orchestrator.py — Updated with Push-to-Talk Button Support

Changes:
  1. Added global _speech_transcriber reference for button access
  2. Added button_control_thread() for GPIO button handling
  3. Modified speech_thread() to store transcriber instance
  4. Added clean shutdown for transcriber + GPIO
"""

import threading
import time
import sys
import signal

# ── GPIO Button Config ────────────────────────────────────────────────────────
try:
    from gpiozero import Button
    BUTTON_PIN = 17  # BCM numbering
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("[Orchestrator] gpiozero not installed — button control disabled")

# ── Blackboard ────────────────────────────────────────────────────────────────
_speech_ready = threading.Event()
_lock = threading.Lock()
blackboard = {
    # Vision
    "ear": 0.30,
    "vision_ok": False,
    # Vitals
    "heart_rate": None,
    "spo2": None,
    "temperature": None,
    "vitals_ok": False,
    # Speech
    "chest_pain": 0,
    "breathless": 0,
    "speech_ok": False,
    # Meta
    "last_triage_level": None,
}

def read_board():
    with _lock:
        return dict(blackboard)

def write_board(**kwargs):
    with _lock:
        blackboard.update(kwargs)

# ── Shared Transcriber Reference ─────────────────────────────────────────────
# Button handlers will call methods on this instance
_speech_transcriber = None

# ── Config ────────────────────────────────────────────────────────────────────
LAPTOP_IP          = "192.168.137.1"
LAPTOP_PORT        = 5001
POST_ENDPOINT      = f"http://{LAPTOP_IP}:{LAPTOP_PORT}/update_vitals"
INFERENCE_INTERVAL = 0.5

# ── Normalisation ─────────────────────────────────────────────────────────────
RANGES = {
    "ear":   (0.0,   0.45),
    "temp":  (34.0,  42.0),
    "spo2":  (70.0, 100.0),
    "pulse": (40.0, 180.0),
}

def _norm(val, key):
    import numpy as np
    lo, hi = RANGES[key]
    return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

# ── Triage model ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import numpy as np

class _TriageNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 3),
        )
    def forward(self, x):
        return self.net(x)

_model = _TriageNN()
try:
    _model.load_state_dict(torch.load("triage_model.pth", map_location="cpu"))
    _model.eval()
    print("[Orchestrator] Triage model loaded.")
except FileNotFoundError:
    print("[Orchestrator] ERROR: triage_model.pth not found. Run training.py first.")
    sys.exit(1)

LEVEL_LABELS = {0: "CRITICAL", 1: "URGENT", 2: "STABLE"}

def run_inference(board):
    ear   = board.get("ear")   or 0.30
    temp  = board.get("temperature") or 37.0
    spo2  = board.get("spo2")  or 98
    pulse = board.get("heart_rate")  or 80
    cp    = board.get("chest_pain", 0)
    br    = board.get("breathless",  0)

    vec = torch.FloatTensor([[
        _norm(ear,   "ear"),
        float(cp),
        float(br),
        _norm(temp,  "temp"),
        _norm(spo2,  "spo2"),
        _norm(pulse, "pulse"),
    ]])

    with torch.no_grad():
        logits = _model(vec)
        probs  = torch.nn.functional.softmax(logits, dim=1)
        conf, pred = torch.max(probs, 1)

    level = pred.item()
    return level, LEVEL_LABELS[level], round(conf.item() * 100, 1)

# ── HTTP POST ─────────────────────────────────────────────────────────────────
import requests

def post_result(board, level_int, level_str, confidence):
    payload = {
        "timestamp":       time.time(),
        "triage_level":    level_int + 1,
        "triage_label":    level_str,
        "confidence_pct":  confidence,
        "ear":             board.get("ear"),
        "heart_rate_bpm":  board.get("heart_rate"),
        "spo2_percent":    board.get("spo2"),
        "temperature_c":   board.get("temperature"),
        "chest_pain":      board.get("chest_pain", 0),
        "breathless":      board.get("breathless",  0),
    }
    try:
        r = requests.post(POST_ENDPOINT, json=payload, timeout=3)
        print(f"[POST] {level_str} ({confidence}%) → {r.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"[POST] Failed: {e}")

# ── Thread 1: Vision ──────────────────────────────────────────────────────────
def vision_thread():
    try:
        from vision import VisionModule
    except ImportError as e:
        print(f"[Vision] Import failed: {e}")
        return

    vm = VisionModule()
    print("[Vision] Thread started.")

    import cv2
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            mesh = results.multi_face_landmarks[0].landmark
            left_ear  = vm.calculate_ear(mesh, vm.LEFT_EYE)
            right_ear = vm.calculate_ear(mesh, vm.RIGHT_EYE)
            avg_ear   = (left_ear + right_ear) / 2.0
            write_board(ear=avg_ear, vision_ok=True)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.multi_face_landmarks[0],
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
            if avg_ear >= 0.35:
                status, color = "ALERT", (0, 255, 0)
            elif avg_ear >= 0.18:
                status, color = "DROWSY", (0, 165, 255)
            else:
                status, color = "UNRESPONSIVE", (0, 0, 255)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, status, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imshow("Triage Vision", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ── Thread 2: Vitals ──────────────────────────────────────────────────────────
def vitals_thread():
    try:
        import pi_push_vitals as pv
    except ImportError as e:
        print(f"[Vitals] Import failed: {e}")
        return
    print("[Vitals] Thread started.")
    pv.ENABLE_HTTP_POST = False

    while True:
        try:
            now = time.time()
            try:
                pv.m.read_sensor()
                pv.ir_buffer.append(pv.m.ir)
                pv.red_buffer.append(pv.m.red)
                pv.time_buffer.append(now)
            except Exception as e:
                print(f"[Vitals] MAX30100 read error: {e}")
            try:
                reg = pv.MLX90614_REG_OBJECT if pv.USE_OBJECT_TEMP else pv.MLX90614_REG_AMBIENT
                temperature = pv.read_mlx90614_temperature(pv.mlx_bus, reg)
            except Exception:
                temperature = None

            min_samples = int(pv.WINDOW_SECONDS / pv.SAMPLE_RATE)
            if len(pv.ir_buffer) >= min_samples:
                finger, signal_ok, _ = pv.detect_finger_and_signal_quality(pv.ir_buffer, pv.red_buffer)
                if finger and signal_ok:
                    hr, _ = pv.detect_heart_rate(pv.time_buffer, pv.ir_buffer)
                    hr = pv.validate_with_hysteresis(hr, pv.last_valid_hr, pv.MIN_BPM, pv.MAX_BPM, hold_cycles=2)
                    if hr is not None:
                        pv.last_valid_hr = hr
                    spo2_raw, spo2_status = pv.calculate_spo2_ratio(pv.red_buffer, pv.ir_buffer)
                    if spo2_status == "valid":
                        spo2 = pv.validate_with_hysteresis(spo2_raw, pv.last_valid_spo2, pv.SPO2_LOW_THRESHOLD, pv.SPO2_HIGH_THRESHOLD)
                    elif spo2_status == "low_confidence":
                        spo2 = pv.validate_with_hysteresis(spo2_raw, pv.last_valid_spo2, 70, 100)
                    else:
                        spo2 = None
                    if spo2 is not None:
                        pv.last_valid_spo2 = spo2
                    write_board(
                        heart_rate=hr if hr is not None else pv.last_valid_hr,
                        spo2=spo2 if spo2 is not None else pv.last_valid_spo2,
                        temperature=temperature,
                        vitals_ok=True,
                    )
            elapsed = time.time() - now
            time.sleep(max(0, pv.SAMPLE_RATE - elapsed))
        except Exception as e:
            print(f"[Vitals] Loop error: {e}")
            time.sleep(1)

# ── Thread 3: Speech (Updated) ────────────────────────────────────────────────
def speech_thread():
    """
    Starts the IntegratedTranscriber and stores the instance globally
    so the button handler can control it.
    """
    global _speech_transcriber
    try:
        from transcriber_integrated2 import IntegratedTranscriber
    except ImportError as e:
        print(f"[Speech] Import failed: {e}")
        return

    _speech_transcriber = IntegratedTranscriber(blackboard_writer=write_board)
    _speech_transcriber.start()  # Returns immediately; runs in background thread
    _speech_ready.set()
    print("[Speech] Thread started.")

    # Keep thread alive (transcriber runs independently)
    while True:
        time.sleep(1)

# ── Button Control Thread (New) ───────────────────────────────────────────────
def button_control_thread():
    """
    Waits for GPIO button events and controls speech transcriber.
    Runs in its own daemon thread.
    """
    if not GPIO_AVAILABLE:
        return
    
    print(f"[Button] Initializing GPIO button on pin {BUTTON_PIN}...")
    button = Button(BUTTON_PIN, pull_up=True)
    
    def on_press():
        if _speech_ready.is_set() and _speech_transcriber:
            _speech_transcriber.enable()
            print("[Button] Speech ENABLED")
        else:
            print("[Button] Speech not ready yet...")
    
    def on_release():
        global _speech_transcriber
        if _speech_transcriber:
            _speech_transcriber.disable()
            print("[Button] Speech DISABLED (processing...)")
    
    button.when_pressed = on_press
    button.when_released = on_release
    print(f"[Button] Ready. Hold button to speak.")
    
    # Keep thread alive
    while True:
        time.sleep(1)

# ── Thread 4: Inference loop ──────────────────────────────────────────────────
def inference_loop():
    print("[Inference] Loop started.")
    while True:
        time.sleep(INFERENCE_INTERVAL)
        board = read_board()
        level_int, level_str, confidence = run_inference(board)
        print(
            f"[Triage] {level_str} ({confidence}%) | "
            f"EAR={board.get('ear', 0):.2f} "
            f"HR={board.get('heart_rate')} "
            f"SpO2={board.get('spo2')} "
            f"Temp={board.get('temperature')} "
            f"CP={board.get('chest_pain', 0)} "
            f"BR={board.get('breathless', 0)}"
        )
        # Always POST every cycle (adjust logic if you want change-only)
        write_board(last_triage_level=level_int)
        post_result(board, level_int, level_str, confidence)

# ── Graceful Shutdown ─────────────────────────────────────────────────────────
def _shutdown(sig, frame):
    print("\n[Orchestrator] Shutdown signal received.")
    
    # Stop transcriber if it exists
    global _speech_transcriber
    if _speech_transcriber:
        print("[Orchestrator] Stopping speech transcriber...")
        _speech_transcriber.stop()
    
    # Clean up GPIO
    if GPIO_AVAILABLE:
        try:
            from gpiozero import Button
            # gpiozero handles cleanup automatically, but we can close explicitly
            print("[Orchestrator] GPIO cleanup complete.")
        except:
            pass
    
    print("[Orchestrator] Goodbye.")
    sys.exit(0)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(" Pi 5 Triage System — Orchestrator + Push-to-Talk")
    print(f" Posting to: {POST_ENDPOINT}")
    print(f" Inference every {INFERENCE_INTERVAL}s")
    if GPIO_AVAILABLE:
        print(f" Button control: GPIO{BUTTON_PIN} (hold to speak)")
    else:
        print(" Button control: DISABLED (install gpiozero to enable)")
    print("=" * 60)

    # Register signal handlers
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # Start threads
    threads = [
        threading.Thread(target=vision_thread,      name="Vision",    daemon=True),
        threading.Thread(target=vitals_thread,      name="Vitals",    daemon=True),
        threading.Thread(target=speech_thread,      name="Speech",    daemon=True),
        threading.Thread(target=inference_loop,     name="Inference", daemon=True),
    ]
    
    # Add button thread if GPIO available
    if GPIO_AVAILABLE:
        threads.append(threading.Thread(target=button_control_thread, name="Button", daemon=True))

    for t in threads:
        t.start()
        print(f"[Orchestrator] Started: {t.name}")

    # Keep main thread alive
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
