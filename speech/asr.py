import io
import numpy as np
from faster_whisper import WhisperModel

class ASRProcessor:
    """
    ASRProcessor wraps the faster-whisper library for efficient speech-to-text.
    Defaulting to the 'tiny' model for edge performance, it can be easily
    swapped for larger models (base, small, medium, large-v3).
    """

    def __init__(self, model_size: str = "tiny", device: str = "cpu", compute_type: str = "int8"):
        """
        Initialize the Whisper model.
        :param model_size: 'tiny', 'base', 'small', 'medium', 'large-v3'
        :param device: 'cpu' or 'cuda'
        :param compute_type: 'int8', 'float16', etc. (int8 is efficient for CPU)
        """
        print(f"Loading ASR Model: {model_size} on {device}...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("ASR Model loaded successfully.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribe a chunk of audio.
        :param audio_data: Numpy array of audio samples (PCM float32, 16kHz)
        :return: Transcribed text string
        """
        segments, info = self.model.transcribe(audio_data, beam_size=5)
        
        # Combine segments into a single string
        transcription = " ".join([segment.text for segment in segments]).strip()
        return transcription

if __name__ == "__main__":
    # Quick test if run as script (requires an audio file or mock data)
    processor = ASRProcessor()
    print("ASR Processor ready.")
