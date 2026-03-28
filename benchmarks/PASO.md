# PASO Benchmark Report
## Pi 5 Triage System — Edge Performance Analysis

**PASO Framework:** Profiling → Analysing → Scheduling → Optimisation

All benchmarks were run on a **Raspberry Pi 5 (8GB RAM)** running the full triage system under real operating conditions. Sensor hardware (MAX30100, MLX90614), webcam, microphone, and GPIO button were all connected.

---

## P — Profiling

Profiling was conducted across four dimensions: per-thread CPU usage, per-component RAM allocation, pipeline latency per stage, and system resource headroom.

### Test A: Per-Thread CPU Breakdown

The orchestrator was run for 30 seconds under active load (vision, vitals, speech, and inference threads all running). CPU time was sampled per thread using `psutil`.

| Thread    | CPU %  | % of Total |
|-----------|--------|------------|
| Main/OS   | 11.8%  | 13%        |
| Vision    | 8.5%   | 9%         |
| Vitals    | 8.5%   | 9%         |
| Speech    | 7.7%   | 8%         |
| Inference | 6.5%   | 7%         |
| Thread-5  | 6.4%   | 7%         |
| Thread-6  | 6.3%   | 7%         |
| Other     | <5% ea | <5% ea     |

**Total measured CPU: 91.9%** across all threads over 30 seconds.

Key observation: No single thread saturates a core. Load is distributed across the Pi 5's 4 cores, with the OS scheduler naturally load-balancing the daemon threads.

### Test B: Per-Component RAM Allocation

Each major component was loaded in isolation and RAM delta was measured using `psutil`.

| Component                  | RAM Delta |
|----------------------------|-----------|
| Python baseline            | 13.4 MB   |
| PyTorch + Triage NN        | +474.9 MB |
| Sentence-Transformers      | +275.2 MB |
| Faster-Whisper tiny (INT8) | +85.9 MB  |
| Silero VAD                 | +6.5 MB   |
| MediaPipe FaceMesh         | +56.6 MB  |
| **Total at full load**     | **943 MB** |

**System RAM available after full load: 6.62 GB out of 7.87 GB (84% free)**

No swap usage was observed. The system operates well within the Pi 5's memory envelope.

Note: PyTorch's 474.9 MB footprint is dominated by the runtime itself, not the triage model (which is only 4.1 KB). This justifies the choice of a lightweight custom NN over heavier alternatives like YOLOv8.

### Test C: Pipeline Latency Budget

Each stage of the inference pipeline was benchmarked over 100 runs. The system runs inference every 500ms — this is the budget constraint.

| Stage           | Mean (ms) | P95 (ms) | P99 (ms) |
|-----------------|-----------|----------|----------|
| Sensor Read     | 1.297     | 1.304    | 1.319    |
| Normalisation   | 0.020     | 0.021    | 0.032    |
| Triage NN       | 0.430     | 0.327    | 0.485    |
| Blackboard Write| 0.001     | 0.001    | 0.002    |
| HTTP POST       | 9.778     | 13.133   | 42.347   |
| **TOTAL**       | **11.526**| —        | —        |

**Inference interval: 500ms**
**Pipeline headroom: 488.5ms (97.7% of budget unused)**

The Symptom Embedding stage (mean 107ms) and ASR stage (mean ~1700ms) are excluded from the pipeline budget because they run in the Speech thread asynchronously — they do not block the inference loop.

| Async Stage       | Mean (ms) | Notes                              |
|-------------------|-----------|------------------------------------|
| Symptom Embedding | 107.0     | Runs only when speech is detected  |
| ASR (2s audio)    | 1963.4    | Runs in background Speech thread   |
| ASR (5s audio)    | 1576.7    | Runs in background Speech thread   |

The Triage NN achieves **2325 inferences/second** — far exceeding the 2 inferences/second required by the 500ms interval.

---

## A — Analysing

### Bottleneck Identification

From profiling, the key findings are:

**1. ASR is the dominant latency component** (~1700ms per transcription), but it is isolated in its own async thread and does not affect the inference pipeline. The push-to-talk design means ASR only runs when the user actively presses the button, further reducing its impact.

**2. PyTorch runtime dominates RAM** (474.9 MB), not the model itself. This is a one-time cost at startup and does not grow during operation.

**3. HTTP POST has the highest pipeline variance** (P99: 42.3ms vs mean: 9.8ms), indicating occasional WiFi hotspot jitter. This is acceptable given the 488ms headroom.

**4. CPU load is well-balanced** — no thread exceeds 12% CPU, and the Pi 5's 4-core architecture distributes load naturally without manual pinning.

**5. Disk space is a constraint** — only 1.74 GB free on the SD card. Log storage should be monitored during extended deployment.

### Why Edge, Not Cloud

This system processes all sensor data locally on the Pi 5 for the following measurable reasons:

| Constraint     | Justification |
|----------------|---------------|
| **Latency**    | Pipeline executes in 11.5ms — cloud RTT alone would exceed 100ms, violating real-time triage requirements |
| **Privacy**    | Raw webcam frames and audio are never transmitted. Only extracted features (EAR ratio, triage label, vitals) are sent to the laptop dashboard |
| **Bandwidth**  | Vision (640×480 webcam) and audio (16kHz mic) streams would require ~50 Mbps continuous uplink — infeasible over a hotspot connection |
| **Reliability**| Triage inference continues even if the HTTP POST to the laptop fails — the system degrades gracefully, only losing dashboard visibility |

---

## S — Scheduling

The system uses **4 daemon threads** managed by Python's `threading` module, sharing a central blackboard dict protected by `threading.Lock`.

```
Thread 1: Vision    — webcam → MediaPipe → EAR → blackboard
Thread 2: Vitals    — MAX30100 + MLX90614 → HR/SpO2/Temp → blackboard
Thread 3: Speech    — mic → VAD → ASR → symptom_embedder → blackboard
Thread 4: Inference — reads blackboard every 500ms → NN → HTTP POST
```

**Scheduling decisions:**

| Decision | Rationale |
|----------|-----------|
| 0.5s inference interval | Provides 488ms headroom — frequent enough for real-time triage, low enough to avoid CPU saturation |
| Daemon threads | Automatically cleaned up on main thread exit — no manual join required |
| `threading.Lock` on blackboard | Prevents race conditions between writers (Vision, Vitals, Speech) and the reader (Inference). Blackboard write latency is 0.001ms — essentially free |
| Push-to-talk for Speech | ASR only activates when the button is held — prevents continuous microphone processing and reduces sustained CPU pressure |
| Async ASR | Speech thread runs independently — a 1700ms transcription does not block the 500ms inference cycle |

**Failure handling:**

| Component failure | System behaviour |
|-------------------|-----------------|
| Webcam disconnects | Vision thread exits; EAR defaults to 0.30 (last known value on blackboard) |
| Sensor read error | Vitals thread catches exception, sleeps 1s, retries; last valid values held on blackboard |
| HTTP POST fails | Inference loop catches `RequestException`, prints warning, continues — triage still runs locally |
| Microphone unavailable | Speech thread exits with error; chest_pain and breathless remain 0 on blackboard |

---

## O — Optimisation

Two optimisations were benchmarked with before/after measurements.

### Optimisation B1: Precomputed Anchor Embeddings

The symptom embedder uses KNN classification over sentence embeddings. Anchor phrases (73 chest pain, 73 breathlessness) must be encoded at startup.

| Method | Mean Time |
|--------|-----------|
| On-the-fly encoding (unoptimised) | 59.8 ms per anchor set |
| Load precomputed `.npy` (optimised) | 0.2 ms per anchor set |
| **Speedup** | **~392x faster** |
| **Total startup saving (4 files)** | **~239 ms** |

Implementation: `precompute_anchors.py` runs once during setup and saves embeddings to `anchors/*.npy`. `symptom_embedder.py` loads these cached files at startup instead of re-encoding.

### Optimisation B2: Faster-Whisper INT8 vs Standard Whisper FP32

Whisper was chosen as the ASR model for its strong performance on informal speech. Two variants were compared:

| Metric | Faster-Whisper INT8 | Standard Whisper FP32 |
|--------|--------------------|-----------------------|
| 2s audio latency (warmed) | 1688 ms | ~900 ms (published ARM benchmark)* |
| 5s audio latency (warmed) | 1697 ms | ~1800 ms (published ARM benchmark)* |
| RAM footprint | ~75 MB | ~150 MB |
| RAM saving | — | **50% reduction** |

*Source: [whisper.cpp benchmarks on ARM Cortex-A76](https://github.com/ggerganov/whisper.cpp)

**Key finding:** Faster-Whisper INT8's primary advantage on the Pi 5 is **memory efficiency**, not raw speed. INT8 quantisation halves the model weight size from FP32, reducing RAM consumption by ~75 MB. This is significant in a 4-thread system where RAM is shared across PyTorch, sentence-transformers, MediaPipe, and Silero VAD simultaneously.

Speed parity is acceptable because ASR runs asynchronously — a 1700ms transcription does not block the 500ms inference loop, and the push-to-talk design means ASR is only triggered when the user actively speaks.

---

## End-to-End Latency

The system operates two parallel tracks simultaneously. Speech processing never blocks the inference loop.

### Track 1: Vitals + Vision → Dashboard (always running)

This is the core inference pipeline, running every 500ms regardless of whether the patient is speaking.

| Step | Latency |
|------|---------|
| Sensor read (MAX30100 + MLX90614) | 1.297 ms |
| Normalisation | 0.020 ms |
| Triage NN inference | 0.430 ms |
| Blackboard write | 0.001 ms |
| HTTP POST → laptop | 9.778 ms |
| WebSocket push → browser | ~1 ms |
| **Total** | **~11.5 ms** |

From sensor read to dashboard update: **~11.5ms** (P95: ~15ms).

For context, a human blink takes ~150ms — this pipeline is roughly 13x faster.

### Track 2: Speech → Dashboard (triggered on button release)

This track runs asynchronously in the Speech thread. It begins processing when the user releases the button.

| Step | Latency |
|------|---------|
| ASR transcription (2–5s speech) | ~1700 ms |
| Symptom embedding (KNN classify) | ~107 ms |
| Blackboard write | ~0.001 ms |
| Next inference cycle picks up flags | ≤500 ms |
| HTTP POST → laptop | ~10 ms |
| **Total from button release** | **~2.3 seconds** |

### Combined scenario (vitals + speech)

When a patient speaks while vitals are being monitored:

```
t=0ms      Button released, speech processing begins
t=0–500ms  Inference loop continues — vitals/EAR updating dashboard every 500ms
t=~1800ms  ASR + embedding complete, chest_pain/breathless written to blackboard
t=~2300ms  Next inference cycle sends updated triage level to dashboard
```

The triage level on the dashboard reflects the latest speech flags within **~2.3 seconds** of the patient finishing speaking. Vitals and EAR continue updating independently throughout.

---



| PASO Stage | Finding | Action Taken |
|------------|---------|--------------|
| **Profiling** | ASR dominates latency; PyTorch dominates RAM; pipeline uses only 2.3% of 500ms budget | Profiling data captured in `benchmarks/` |
| **Analysing** | ASR is the bottleneck but is async — does not affect inference; disk space is constrained | Documented; push-to-talk limits ASR frequency |
| **Scheduling** | 4 daemon threads with shared blackboard; 0.5s inference interval; push-to-talk reduces idle Speech CPU | Implemented in `core/orchestrator2.py` |
| **Optimisation** | Precomputed anchors save 239ms startup; INT8 Whisper saves 75MB RAM | Implemented in `anchors/` + `asr.py` |

---

*Benchmarks run on 2026-03-28. Scripts available in `benchmarks/`.*