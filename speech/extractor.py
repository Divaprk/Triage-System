import spacy
import scispacy
from typing import List, Dict

class SymptomExtractor:
    """
    SymptomExtractor extracts clinical concepts from transcribed medical speech.
    Using scispacy's 'en_core_sci_sm' model, it focuses on medical entities
    like symptoms, anatomical locations, and diseases.
    """

    def __init__(self, model_name: str = "en_core_sci_sm"):
        """
        Initialize the NLP pipeline.
        :param model_name: scispacy model name (e.g. 'en_core_sci_sm')
        """
        print(f"Loading Medical Entity Model: {model_name}...")
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Please run the install script or download manually.")
            raise ImportError(f"Model '{model_name}' not installed. Download with: 'pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz'")
        print("Medical NER Model loaded successfully.")

    def extract(self, text: str) -> List[Dict[str, str]]:
        """
        Extract symptoms and medical entities from text.
        :param text: Transcribed clinical speech
        :return: List of extracted entities with labels
        """
        if not text:
            return []

        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            # ent.label_ might be 'ENTITY' for general concepts in small model
            # but we'll return its text and start/end positions for reference
            entities.append({
                "entity": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        return entities

if __name__ == "__main__":
    # Test text
    sample_text = "Patient complains of chest pain, shortness of breath and a persistent cough since last Tuesday."
    
    try:
        extractor = SymptomExtractor()
        results = extractor.extract(sample_text)
        print(f"Extracted Symptoms: {results}")
    except Exception as e:
        print(f"Extraction failed: {e}")
