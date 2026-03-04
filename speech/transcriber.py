import os
import queue
import threading
import torch
import numpy as np
import sounddevice as sd
from asr import ASRProcessor
from extractor import SymptomExtractor
import time

# VAD and Audio Configuration
# silero-vad expects 16,000 Hz or 8,000 Hz
SAMPLE_RATE = 16000
# Duration of audio chunks in ms for VAD
WINDOW_SIZE_SAMPLES = 512 # approx 32ms at 16kHz
VAD_THRESHOLD = 0.5 

class RealTimeTranscriber:
    """
    RealTimeTranscriber orchestrates audio capture, VAD (Voice Activity Detection),
    ASR (Automatic Speech Recognition), and NER (Named Entity Recognition).
    It manages non-blocking audio capture and buffers chunks until silence is detected.
    """

    def __init__(self, asr_model="tiny", ner_model="en_core_sci_sm"):
        """
        Initialize the transcription pipeline.
        """
        print("Initializing MedExtract Orchestrator...")
        
        # Load models
        self.asr = ASRProcessor(model_size=asr_model)
        self.ner = SymptomExtractor(model_name=ner_model)
        
        # Load silero-vad model
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                                model='silero_vad',
                                                force_reload=False,
                                                onnx=False)
        (self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks) = utils
        
        # Buffers and state
        self.audio_queue = queue.Queue()
        self.is_running = False
        self.audio_buffer = [] # To accumulate speech chunks
        self.silence_counter = 0 # To detect a 'natural pause' (e.g. 1 second)
        self.PAUSE_LIMIT = 30 # Number of silent windows (~1 sec) before transcribing

    def audio_callback(self, indata, frames, time, status):
        """
        Callback for sounddevice. Indata is a numpy array.
        """
        if status:
            print(f"Audio Status Error: {status}")
        self.audio_queue.put(indata.copy())

    def process_loop(self):
        """
        Main loop to process audio and handle VAD triggers.
        """
        print("\n[READY] Listening to microphone... Speaking will trigger transcription.")
        while self.is_running:
            try:
                # Grab a chunk of audio from the queue (non-blocking)
                chunk = self.audio_queue.get(timeout=1.0)
                
                # Check for voice activity
                # silero-vad requires audio to be torch tensor (normalized float32)
                tensor_chunk = torch.from_numpy(chunk.squeeze().astype(np.float32))
                speech_prob = self.vad_model(tensor_chunk, SAMPLE_RATE).item()
                
                if speech_prob > VAD_THRESHOLD:
                    # Speech detected
                    self.audio_buffer.append(chunk)
                    self.silence_counter = 0
                else:
                    # Silence detected
                    if len(self.audio_buffer) > 0:
                        self.silence_counter += 1
                        
                        # If silence persists, finalize transcription
                        if self.silence_counter >= self.PAUSE_LIMIT:
                            print("\n[TRANSCRIPT-TRIGGER] Natural pause detected. Processing chunk...")
                            self.finalize_segment()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in process loop: {e}")

    def finalize_segment(self):
        """
        Convert buffered audio to text, extract symptoms, and clear buffer.
        """
        if not self.audio_buffer:
            return

        # Concatenate audio into a single numpy array
        full_audio = np.concatenate(self.audio_buffer).squeeze()
        
        # Step 1: ASR
        text = self.asr.transcribe(full_audio)
        if text:
            print(f"User: {text}")
            
            # Step 2: Symptom Extraction (NER)
            symptoms = self.ner.extract(text)
            
            # Print extracted symptoms
            if symptoms:
                print(f"Extracted Symptoms: {', '.join([s['entity'] for s in symptoms])}")
            else:
                print("No symptoms detected.")
        else:
            print("No speech recognized in the segment.")
            
        print("-" * 20)
        
        # Reset buffers
        self.audio_buffer = []
        self.silence_counter = 0

    def start(self):
        """
        Start the non-blocking record thread and processing loop.
        """
        self.is_running = True
        
        # Start sounddevice stream
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, 
                                     channels=1, 
                                     callback=self.audio_callback,
                                     blocksize=WINDOW_SIZE_SAMPLES)
        
        with self.stream:
            try:
                self.process_loop()
            except KeyboardInterrupt:
                print("\n[INFO] MedExtract stopping...")
                self.is_running = False

if __name__ == "__main__":
    orchestrator = RealTimeTranscriber()
    orchestrator.start()
