import logging
from typing import Tuple, Dict, Optional
import warnings

# Suppress fastcoref warnings if needed
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

try:
    from fastcoref import FCoref
    HAS_FASTCOREF = True
except ImportError:
    HAS_FASTCOREF = False
    logger.warning("fastcoref not installed. Coreference resolution will be disabled.")

class CorefResolver:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CorefResolver, cls).__new__(cls)
            cls._instance.model = None
        return cls._instance

    def load_model(self, model_name: str = "fastcoref"):
        if not HAS_FASTCOREF:
            return
            
        if self.model is None:
            logger.info(f"Loading coreference model: {model_name}")
            try:
                # FCoref defaults to a small efficient model
                self.model = FCoref(device='cpu') # Use CPU by default for stability, or check config
                logger.info("Coreference model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load coreference model: {e}")
                self.model = None

    def resolve(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Resolve pronouns in text.
        Returns:
            resolved_text: Text with pronouns replaced (or just original if no changes)
            coref_map: Dictionary mapping {pronoun_text: antecedent_text}
        """
        if not HAS_FASTCOREF:
            return text, {}
            
        if self.model is None:
            self.load_model()
            
        if self.model is None:
            return text, {}

        try:
            preds = self.model.predict(texts=[text])
            
            # fastcoref returns a list of predictions
            pred = preds[0]
            
            clusters = pred.get_clusters(as_strings=True)
            # clusters is a list of list of strings, e.g. [['The engine', 'It'], ['John', 'he']]
            # We want to map pronouns to the most specific antecedent (usually the first mention)
            
            coref_map = {}
            
            # Simple strategy: Map all subsequent mentions to the first mention in the cluster
            for cluster in clusters:
                if not cluster or len(cluster) < 2:
                    continue
                    
                antecedent = cluster[0]
                for mention in cluster[1:]:
                    # Filter for likely pronouns or deictics to avoid over-resolving
                    # But fastcoref usually handles this. 
                    # User asked for "pronoun/deictic resolution".
                    # Let's map everything for now as requested.
                    if mention.lower() != antecedent.lower():
                        coref_map[mention] = antecedent
            
            # Get resolved text directly from fastcoref if available, 
            # but FCoref.predict doesn't return resolved text directly in the simple API?
            # Actually it does not. We have to reconstruct it or use `get_clusters`.
            # Wait, the user prompt example showed: `result["text"]` which implies some API.
            # But standard FCoref returns FCorefResult.
            # Let's check if there's a utility or if we need to replace manually.
            
            # Manual replacement strategy (naive but works for now):
            # We need span indices to do this correctly.
            # pred.get_clusters(as_strings=False) returns spans.
            
            clusters_spans = pred.get_clusters(as_strings=False)
            # clusters_spans: List[List[Tuple[int, int]]]
            
            # We need to replace from end to start to avoid messing up indices
            replacements = []
            
            for cluster_idx, cluster in enumerate(clusters_spans):
                # Get antecedent text
                first_span = cluster[0]
                antecedent_text = text[first_span[0]:first_span[1]]
                
                # For subsequent mentions
                for span in cluster[1:]:
                    start, end = span
                    mention_text = text[start:end]
                    
                    # Only replace if it looks like a pronoun/reference
                    # (Optional safety check, but let's trust the model for now)
                    replacements.append((start, end, antecedent_text))
                    
                    # Populate map
                    coref_map[mention_text] = antecedent_text
            
            # Sort replacements by start index descending
            replacements.sort(key=lambda x: x[0], reverse=True)
            
            resolved_text = text
            for start, end, replacement in replacements:
                resolved_text = resolved_text[:start] + replacement + resolved_text[end:]
                
            return resolved_text, coref_map
            
        except Exception as e:
            logger.error(f"Error during coreference resolution: {e}")
            return text, {}

# Global instance
resolver = CorefResolver()

def resolve(text: str) -> Tuple[str, Dict[str, str]]:
    return resolver.resolve(text)
