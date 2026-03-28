"""
latency_budget.py — Pipeline Latency Budget Benchmark

Instruments each stage of the triage pipeline end-to-end:
  Stage 1 : Sensor read       (MAX30100 + MLX90614 I2C read)
  Stage 2 : Normalisation     (feature scaling to [0,1])
  Stage 3 : Triage NN         (6-feature MLP inference)
  Stage 4 : Blackboard write  (threading.Lock + dict update)
  Stage 5 : HTTP POST         (JSON payload to laptop endpoint)
  Stage 6 : Symptom embedding (sentence-transformers KNN classify)
  Stage 7 : ASR               (Faster-Whisper tiny INT8, 2s and 5s audio)

Runs each stage 100 times (except HTTP POST: 20 times) and reports:
  - Mean, min, max, p95, p99 latency
  - Total pipeline budget (excluding ASR, which is async)
  - Comparison to 500ms inference interval budget

Usage (run from triage_system/ root):
    cd ~/triage_system
    source venv/bin/activate
    python3.11 benchmarks/latency_budget.py

Output:
    benchmarks/latency_budget_report.txt
    benchmarks/latency_budget_raw.json
"""

import time
import json
import threading
import numpy as np
from datetime import datetime

RUNS = 100
HTTP_RUNS = 20
REPORT_FILE = "benchmarks/latency_budget_report.txt"
RAW_FILE    = "benchmarks/latency_budget_raw.json"

results = {}

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def log(msg=""):
    print(f"  {msg}")

def stats(times_ms):
    a = np.array(times_ms)
    return {
        "mean_ms":  round(float(np.mean(a)),   3),
        "min_ms":   round(float(np.min(a)),    3),
        "max_ms":   round(float(np.max(a)),    3),
        "p95_ms":   round(float(np.percentile(a, 95)), 3),
        "p99_ms":   round(float(np.percentile(a, 99)), 3),
    }

# ── Stage 1: Sensor Read (simulated if hardware absent) ───────
def bench_sensor_read():
    section("Stage 1: Sensor Read")
    log("Attempting real MAX30100 + MLX90614 read...")
    log("(Falls back to I2C timing simulation if hardware absent)")

    times = []
    hardware_available = False

    try:
        import smbus2
        import sys
        sys.path.insert(0, "core")
        from max30100 import MAX30100

        m = MAX30100()
        m.enable_spo2()

        bus = smbus2.SMBus(1)
        MLX90614_ADDR = 0x5A
        MLX90614_REG  = 0x07

        for _ in range(RUNS):
            t0 = time.perf_counter()
            m.read_sensor()
            data = bus.read_i2c_block_data(MLX90614_ADDR, MLX90614_REG, 3)
            times.append((time.perf_counter() - t0) * 1000)

        hardware_available = True
        log(f"Real hardware used ({RUNS} runs)")

    except Exception as e:
        log(f"Hardware not available ({e})")
        log("Simulating I2C read timing (2 x read_i2c_block_data equivalent)...")

        # Simulate realistic I2C overhead: 4-byte + 3-byte read at 100kHz
        for _ in range(RUNS):
            t0 = time.perf_counter()
            # Simulate I2C bus timing: ~0.4ms for 7 bytes at 100kHz
            time.sleep(0.0004)
            times.append((time.perf_counter() - t0) * 1000)

    s = stats(times)
    s["hardware"] = hardware_available
    results["stage1_sensor_read"] = s
    log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | P99: {s['p99_ms']} ms")

# ── Stage 2: Normalisation ────────────────────────────────────
def bench_normalisation():
    section("Stage 2: Normalisation (feature scaling)")

    RANGES = {
        "ear":   (0.0,   0.45),
        "temp":  (34.0,  42.0),
        "spo2":  (70.0, 100.0),
        "pulse": (40.0, 180.0),
    }

    def normalise(val, lo, hi):
        return float(np.clip((val - lo) / (hi - lo), 0.0, 1.0))

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        n_ear   = normalise(0.35, *RANGES["ear"])
        n_temp  = normalise(37.2, *RANGES["temp"])
        n_spo2  = normalise(96,   *RANGES["spo2"])
        n_pulse = normalise(82,   *RANGES["pulse"])
        _ = [n_ear, 0, 0, n_temp, n_spo2, n_pulse]
        times.append((time.perf_counter() - t0) * 1000)

    s = stats(times)
    results["stage2_normalisation"] = s
    log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | P99: {s['p99_ms']} ms")

# ── Stage 3: Triage NN Inference ──────────────────────────────
def bench_triage_nn():
    section("Stage 3: Triage NN Inference")

    try:
        import torch
        import torch.nn as nn

        class TriageNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(6, 16), nn.ReLU(),
                    nn.Linear(16, 8), nn.ReLU(),
                    nn.Linear(8, 3),
                )
            def forward(self, x):
                return self.net(x)

        model = TriageNN()
        model.load_state_dict(torch.load("triage_model.pth", map_location="cpu"))
        model.eval()

        vec = torch.FloatTensor([[0.78, 0.0, 0.0, 0.40, 0.87, 0.30]])

        # Warm up
        with torch.no_grad():
            _ = model(vec)

        times = []
        for _ in range(RUNS):
            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(vec)
                probs  = torch.nn.functional.softmax(logits, dim=1)
                conf, pred = torch.max(probs, 1)
            times.append((time.perf_counter() - t0) * 1000)

        s = stats(times)
        s["throughput_per_sec"] = round(1000 / s["mean_ms"], 1)
        results["stage3_triage_nn"] = s
        log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | "
            f"Throughput: {s['throughput_per_sec']} inferences/s")

    except Exception as e:
        log(f"ERROR: {e}")
        results["stage3_triage_nn"] = {"error": str(e)}

# ── Stage 4: Blackboard Write ─────────────────────────────────
def bench_blackboard_write():
    section("Stage 4: Blackboard Write (threading.Lock)")

    _lock = threading.Lock()
    blackboard = {
        "ear": 0.35, "heart_rate": 82, "spo2": 96,
        "temperature": 37.2, "chest_pain": 0, "breathless": 0,
        "last_triage_level": None,
    }

    def write(**kwargs):
        with _lock:
            blackboard.update(kwargs)

    times = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        write(ear=0.35, heart_rate=82, spo2=96,
              temperature=37.2, last_triage_level=2)
        times.append((time.perf_counter() - t0) * 1000)

    s = stats(times)
    results["stage4_blackboard_write"] = s
    log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | P99: {s['p99_ms']} ms")

# ── Stage 5: HTTP POST ────────────────────────────────────────
def bench_http_post():
    section("Stage 5: HTTP POST to Laptop")
    log(f"Sending {HTTP_RUNS} POST requests to 192.168.137.1:5001...")
    log("(Measures network + Flask round-trip latency)")

    try:
        import requests

        payload = {
            "timestamp":      time.time(),
            "triage_level":   3,
            "triage_label":   "STABLE",
            "confidence_pct": 100.0,
            "ear":            0.35,
            "heart_rate_bpm": 82,
            "spo2_percent":   96,
            "temperature_c":  37.2,
            "chest_pain":     0,
            "breathless":     0,
        }

        times = []
        failures = 0

        for _ in range(HTTP_RUNS):
            t0 = time.perf_counter()
            try:
                r = requests.post(
                    "http://192.168.137.1:5001/update_vitals",
                    json=payload, timeout=3
                )
                elapsed = (time.perf_counter() - t0) * 1000
                if r.status_code == 200:
                    times.append(elapsed)
                else:
                    failures += 1
            except Exception:
                failures += 1

        if times:
            s = stats(times)
            s["successful_posts"] = len(times)
            s["failed_posts"] = failures
            results["stage5_http_post"] = s
            log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | "
                f"Failures: {failures}/{HTTP_RUNS}")
        else:
            log("All POST requests failed — laptop not reachable.")
            log("Recording timeout fallback value (3000ms).")
            results["stage5_http_post"] = {
                "mean_ms": 3000.0, "min_ms": 3000.0, "max_ms": 3000.0,
                "p95_ms": 3000.0, "p99_ms": 3000.0,
                "successful_posts": 0, "failed_posts": HTTP_RUNS,
                "note": "laptop unreachable — timeout fallback recorded"
            }

    except ImportError:
        log("requests not available")
        results["stage5_http_post"] = {"error": "requests not installed"}

# ── Stage 6: Symptom Embedding ───────────────────────────────
def bench_symptom_embedding():
    section("Stage 6: Symptom Embedding (KNN classify)")

    try:
        import sys
        sys.path.insert(0, "core")
        from symptom_embedder import detect_symptoms

        test_phrases = [
            "my chest hurts badly",
            "I cannot breathe properly",
            "I feel fine, no issues",
            "shortness of breath",
            "chest pressure and tightness",
        ]

        # Warm up
        detect_symptoms(test_phrases[0])

        times = []
        for phrase in test_phrases * (RUNS // len(test_phrases)):
            t0 = time.perf_counter()
            _ = detect_symptoms(phrase)
            times.append((time.perf_counter() - t0) * 1000)

        s = stats(times)
        s["phrases_tested"] = len(test_phrases)
        results["stage6_symptom_embedding"] = s
        log(f"Mean: {s['mean_ms']} ms | P95: {s['p95_ms']} ms | P99: {s['p99_ms']} ms")

    except Exception as e:
        log(f"ERROR: {e}")
        results["stage6_symptom_embedding"] = {"error": str(e)}

# ── Stage 7: ASR ──────────────────────────────────────────────
def bench_asr():
    section("Stage 7: ASR (Faster-Whisper tiny INT8)")
    log("Note: ASR runs async — not in the 500ms inference budget.")
    log("Included for reference only.")

    try:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny", device="cpu", compute_type="int8")

        audio_2s = np.random.randn(32000).astype(np.float32) * 0.001
        audio_5s = np.random.randn(80000).astype(np.float32) * 0.001

        ASR_RUNS = 5

        times_2s = []
        for _ in range(ASR_RUNS):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(audio_2s, beam_size=5)
            list(segs)
            times_2s.append((time.perf_counter() - t0) * 1000)

        times_5s = []
        for _ in range(ASR_RUNS):
            t0 = time.perf_counter()
            segs, _ = model.transcribe(audio_5s, beam_size=5)
            list(segs)
            times_5s.append((time.perf_counter() - t0) * 1000)

        s2 = stats(times_2s)
        s5 = stats(times_5s)

        results["stage7_asr"] = {
            "audio_2s": s2,
            "audio_5s": s5,
            "note": "Async thread — excluded from pipeline budget"
        }

        log(f"2s audio: mean {s2['mean_ms']} ms | P95 {s2['p95_ms']} ms")
        log(f"5s audio: mean {s5['mean_ms']} ms | P95 {s5['p95_ms']} ms")
        log("(These run in background — user speaks, button released, then transcribed)")

    except Exception as e:
        log(f"ERROR: {e}")
        results["stage7_asr"] = {"error": str(e)}

# ── Pipeline Budget Summary ───────────────────────────────────
def pipeline_summary():
    section("Pipeline Latency Budget Summary")

    BUDGET_MS = 500  # inference interval

    stages = [
        ("Sensor Read",       "stage1_sensor_read"),
        ("Normalisation",     "stage2_normalisation"),
        ("Triage NN",         "stage3_triage_nn"),
        ("Blackboard Write",  "stage4_blackboard_write"),
        ("HTTP POST",         "stage5_http_post"),
    ]

    total_mean = 0
    rows = []
    for label, key in stages:
        r = results.get(key, {})
        mean = r.get("mean_ms", 0)
        p95  = r.get("p95_ms", 0)
        total_mean += mean
        rows.append((label, mean, p95))

    log(f"{'Stage':<22} {'Mean (ms)':>10} {'P95 (ms)':>10}")
    log("-" * 44)
    for label, mean, p95 in rows:
        log(f"{label:<22} {mean:>10.3f} {p95:>10.3f}")
    log("-" * 44)
    log(f"{'TOTAL (excl. ASR)':<22} {total_mean:>10.3f}")
    log(f"{'Inference interval':<22} {BUDGET_MS:>10}")
    log(f"{'Headroom':<22} {BUDGET_MS - total_mean:>10.3f} ms")
    log()

    asr = results.get("stage7_asr", {})
    if "audio_2s" in asr:
        log(f"ASR (async, 2s audio): {asr['audio_2s']['mean_ms']} ms mean")
        log(f"ASR (async, 5s audio): {asr['audio_5s']['mean_ms']} ms mean")
        log("ASR runs in Speech thread — does not block inference loop.")

    results["pipeline_summary"] = {
        "total_pipeline_mean_ms": round(total_mean, 3),
        "inference_interval_ms": BUDGET_MS,
        "headroom_ms": round(BUDGET_MS - total_mean, 3),
        "stages": [{"stage": l, "mean_ms": m, "p95_ms": p} for l, m, p in rows]
    }

# ── Save Report ───────────────────────────────────────────────
def save_report():
    section("Saving Report")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    r = results
    ps = r.get("pipeline_summary", {})
    stages = ps.get("stages", [])

    report = f"""
Pi 5 Triage System — Pipeline Latency Budget Report
Generated: {timestamp}
{'='*55}

COMPONENT LATENCY BREAKDOWN (100 runs each)

{'Stage':<22} {'Mean (ms)':>10} {'P95 (ms)':>10} {'P99 (ms)':>10}
{'-'*54}
"""
    stage_keys = [
        ("Sensor Read",      "stage1_sensor_read"),
        ("Normalisation",    "stage2_normalisation"),
        ("Triage NN",        "stage3_triage_nn"),
        ("Blackboard Write", "stage4_blackboard_write"),
        ("HTTP POST",        "stage5_http_post"),
        ("Symptom Embed",    "stage6_symptom_embedding"),
    ]
    for label, key in stage_keys:
        d = r.get(key, {})
        report += (f"{label:<22} {d.get('mean_ms', 'N/A'):>10} "
                   f"{d.get('p95_ms', 'N/A'):>10} "
                   f"{d.get('p99_ms', 'N/A'):>10}\n")

    report += f"""
{'='*55}
PIPELINE BUDGET (excluding async ASR)

  Total mean latency : {ps.get('total_pipeline_mean_ms', 'N/A')} ms
  Inference interval : {ps.get('inference_interval_ms', 500)} ms
  Headroom           : {ps.get('headroom_ms', 'N/A')} ms

ASR LATENCY (async — Speech thread, does not block inference)
"""
    asr = r.get("stage7_asr", {})
    if "audio_2s" in asr:
        report += f"  2s audio : {asr['audio_2s']['mean_ms']} ms mean\n"
        report += f"  5s audio : {asr['audio_5s']['mean_ms']} ms mean\n"
        report += f"  Note     : {asr.get('note', '')}\n"

    report += f"\n{'='*55}\n"

    with open(REPORT_FILE, "w") as f:
        f.write(report)

    with open(RAW_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(report)
    print(f"  Report saved: {REPORT_FILE}")
    print(f"  Raw JSON   : {RAW_FILE}")


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nPi 5 Triage System — Pipeline Latency Budget Benchmark")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Runs per stage: {RUNS} (HTTP POST: {HTTP_RUNS})")

    bench_sensor_read()
    bench_normalisation()
    bench_triage_nn()
    bench_blackboard_write()
    bench_http_post()
    bench_symptom_embedding()
    bench_asr()
    pipeline_summary()
    save_report()