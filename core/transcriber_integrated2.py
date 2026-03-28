"""
transcriber_integrated.py

Updated for Button Integration:
  1. Added threading support (runs in background).
  2. Added enable()/disable() methods for Push-to-Talk control.
  3. Audio callback drops data when disabled to save memory/CPU.
  4. Forces finalization of speech buffer when disabled.
"""

import queue
import threading
import torch
import numpy as np
import sounddevice as sd
import time

from asr import ASRProcessor
from symptom_embedder import detect_symptoms

# ── Audio / VAD config ────────────────────────────────────────────────────────

SAMPLE_RATE         = 16000
WINDOW_SIZE_SAMPLES = 512     # ~32 ms at 16 kHz
VAD_THRESHOLD       = 0.5
PAUSE_LIMIT         = 30      # silent windows before finalising (~1 s)
QUEUE_TIMEOUT       = 0.2     # Check for disable signal frequently


class IntegratedTranscriber:
    """
    Mic → Silero VAD → Whisper ASR → symptom_embedder → blackboard.
    
    Now supports background execution and external enable/disable control.
    """

    def __init__(self, asr_model: str = "tiny", blackboard_writer=None):
        print("[Speech] Loading ASR model…")
        self.asr = ASRProcessor(model_size=asr_model)

        print("[Speech] Loading Silero VAD…")
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        (
            self.get_speech_timestamps,
            self.save_audio,
            self.read_audio,
            self.VADIterator,
            self.collect_chunks,
        ) = utils

        self.blackboard_writer = blackboard_writer or (lambda **kw: None)

        self.audio_queue     = queue.Queue()
        self.audio_buffer    = []
        self.silence_counter = 0
        
        # Control Flags
        self.is_running      = False
        self.enabled_event   = threading.Event() # Controls listening state
        self.thread          = None
        self.stream          = None

    # ── sounddevice callback ───────────────────────────────────────────────────

    def _audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[Speech] Audio status: {status}")
        
        # Only process audio if enabled (Push-to-Talk logic)
        if self.enabled_event.is_set():
            self.audio_queue.put(indata.copy())
        # Else: Drop audio to prevent queue buildup when button is not pressed

    # ── VAD + finalize loop ────────────────────────────────────────────────────

    def _process_loop(self):
        print("[Speech] Processing loop started (Waiting for enable signal)…")
        
        while self.is_running:
            # If disabled, check if we need to flush buffer, then wait
            if not self.enabled_event.is_set():
                if self.audio_buffer:
                    print("[Speech] Button released. Finalizing segment…")
                    self._finalize_segment()
                time.sleep(0.1)
                continue

            # Try to get audio with short timeout to remain responsive to disable
            try:
                chunk = self.audio_queue.get(timeout=QUEUE_TIMEOUT)
            except queue.Empty:
                continue

            tensor_chunk = torch.from_numpy(chunk.squeeze().astype(np.float32))
            speech_prob  = self.vad_model(tensor_chunk, SAMPLE_RATE).item()

            if speech_prob > VAD_THRESHOLD:
                self.audio_buffer.append(chunk)
                self.silence_counter = 0
            else:
                if self.audio_buffer:
                    self.silence_counter += 1
                    if self.silence_counter >= PAUSE_LIMIT:
                        self._finalize_segment()

    def _finalize_segment(self):
        if not self.audio_buffer:
            return

        full_audio = np.concatenate(self.audio_buffer).squeeze()

        # Step 1 — ASR
        text = self.asr.transcribe(full_audio)

        if text:
            print(f"[Speech] Transcribed: {text}")

            # Step 2 — symptom_embedder
            result = detect_symptoms(text)

            cp = result["chest_pain"]
            br = result["breathlessness"]

            print(
                f"[Speech] ChestPain={cp} (vote={result['chest_pain_vote']:.2f})  "
                f"Breathless={br} (vote={result['breathlessness_vote']:.2f})"
            )

            # Write to shared blackboard
            self.blackboard_writer(
                chest_pain=cp,
                breathless=br,
                speech_ok=True,
            )
        else:
            print("[Speech] No speech recognised in segment.")

        # Reset buffers
        self.audio_buffer    = []
        self.silence_counter = 0

    # ── Public Control Methods ─────────────────────────────────────────────────

    def start(self):
        """Start the audio stream and processing loop in a background thread."""
        if self.is_running:
            return

        self.is_running = True
        self.enabled_event.clear() # Start in disabled state (wait for button)

        # Initialize Stream (do not use 'with' context manager here to avoid blocking)
        self.stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self._audio_callback,
            blocksize=WINDOW_SIZE_SAMPLES,
        )
        self.stream.start()

        # Start Processing Thread
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()
        print("[Speech] System Ready. Press button to speak.")

    def stop(self):
        """Stop the audio stream and join thread."""
        self.is_running = False
        self.enabled_event.set() # Wake up thread if sleeping
        
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
            
        print("[Speech] System Stopped.")

    def enable(self):
        """Activate listening (Call on Button Press)."""
        if not self.is_running:
            self.start()
        self.enabled_event.set()
        # print("[Debug] Listening Enabled")

    def disable(self):
        """Deactivate listening (Call on Button Release)."""
        self.enabled_event.clear()
        # print("[Debug] Listening Disabled")
