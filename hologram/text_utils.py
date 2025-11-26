import logging
from typing import List
import warnings

# Filter warnings from transformers/gliner if needed
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

# Configure logging
logger = logging.getLogger(__name__)

class ConceptExtractor:
    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1"):
        self.model_name = model_name
        self.model = None
        # Broad set of labels to capture various types of concepts and relations
        self.labels = list(set([
            "concept", 
            "entity", 
            "phenomenon", 
            "object", 
            "theory", 
            "law",
            "definition",
            "property",
            "action",
            "relationship",
            "interaction",
            "verb",
            "relation"
        ]))

    def load_model(self):
        if self.model is None:
            if GLiNER is None:
                logger.error("GLiNER library not installed. Install with 'pip install gliner'")
                return

            logger.info(f"Loading GLiNER model: {self.model_name}")
            try:
                self.model = GLiNER.from_pretrained(self.model_name)
                logger.info("GLiNER model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load GLiNER model: {e}")
                # Don't raise, just log, so we can fallback or handle gracefully
                self.model = None

    def extract_concepts(self, text: str, threshold: float = 0.25) -> List[str]:
        """
        Extract atomic concepts from text using GLiNER.
        Returns a list of unique concept strings, preserving order of appearance.
        """
        if not text or not text.strip():
            return []
        
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            return [text.strip()]
        
        try:
            entities = self.model.predict_entities(text, self.labels, threshold=threshold)
            
            # Sort entities by start position to preserve narrative order
            entities.sort(key=lambda x: x['start'])
            
            # Deduplicate while preserving order
            seen = set()
            concepts = []
            for e in entities:
                text_val = e["text"].strip()
                if text_val not in seen:
                    concepts.append(text_val)
                    seen.add(text_val)
            
            # If no concepts found, maybe the whole text is a concept?
            if not concepts and len(text.split()) < 10:
                 return [text.strip()]
                 
            return concepts
        except Exception as e:
            import traceback
            logger.error(f"Error extracting concepts: {e}\n{traceback.format_exc()}")
            return [text.strip()]

# Global instance
extractor = ConceptExtractor()

def extract_concepts(text: str) -> List[str]:
    return extractor.extract_concepts(text)
