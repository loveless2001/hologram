import logging
import os
from typing import List, Optional, Any
import warnings
import pkg_resources

import json

# Filter warnings from transformers/gliner if needed
warnings.filterwarnings("ignore", category=FutureWarning)

ALIAS_FILE = "data/aliases.json"

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None

try:
    from symspellpy import SymSpell, Verbosity
    HAS_SYM_SPELL = True
except ImportError:
    HAS_SYM_SPELL = False

# Configure logging
logger = logging.getLogger(__name__)

# --- 1. TextCleaner ---
def clean_text(t: str) -> str:
    if not t:
        return ""
    t = t.strip()
    t = t.replace("—", "-")
    t = t.replace("…", "...")
    t = t.replace("\u00a0", " ")  # non-breaking spaces
    return " ".join(t.split())  # collapse multiple spaces

# --- 2. SpellCorrector ---
sym_spell = None
if HAS_SYM_SPELL:
    try:
        sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        logger.info("SymSpell dictionary loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load SymSpell dictionary: {e}")
        sym_spell = None

def correct_spelling(text: str) -> str:
    if not text or sym_spell is None:
        return text
    
    # We process word by word to avoid merging words incorrectly, 
    # though SymSpell supports compound splitting, simple lookup is safer for now.
    out = []
    # Split by space to preserve structure, but this is naive. 
    # Better to just use lookup_compound if we trust it, but user suggested word-by-word.
    # Let's stick to the user's loop for safety and control.
    for word in text.split():
        # Check if word is just punctuation or numbers? 
        # SymSpell handles some, but let's just pass it through.
        suggestions = sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2)
        if suggestions:
            suggestion = suggestions[0].term
            # Log significant changes?
            if suggestion != word and logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Spell correct: {word} -> {suggestion}")
            out.append(suggestion)
        else:
            out.append(word)
    return " ".join(out)

# --- 2.5 Alias Manager ---
_aliases = None
def load_aliases():
    global _aliases
    if _aliases is None:
        if os.path.exists(ALIAS_FILE):
            try:
                with open(ALIAS_FILE, "r", encoding="utf-8") as f:
                    _aliases = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load aliases: {e}")
                _aliases = {}
        else:
            _aliases = {}
    return _aliases

# --- 3. FuzzyResolver & Normalization ---
def normalize_text(text: str, store: Any = None, encoder: Any = None) -> tuple[str, Optional[str]]:
    """
    Apply cleaning, spelling correction, aliasing, and optional fuzzy concept resolution.
    
    Returns:
        tuple: (canonical_text, canonical_trace_id)
            - canonical_text: The normalized/canonical form of the text
            - canonical_trace_id: If fuzzy-resolved, the trace_id of the canonical concept. None otherwise.
    """
    # 1. Clean
    t = clean_text(text)
    
    # 2. Spell Correct
    t_corrected = correct_spelling(t)
    
    # Log correction if changed
    if t_corrected != t:
        logger.info(f"[Correction] '{t}' -> '{t_corrected}'")
        t = t_corrected

    # 2.5 Check Aliases
    aliases = load_aliases()
    if t in aliases:
        canonical = aliases[t]
        logger.info(f"[Alias] '{t}' -> '{canonical}'")
        return (canonical, None)  # Alias, but no trace_id yet

    # 3. Fuzzy Resolution (Concept Canonicalization)
    if store is not None and encoder is not None:
        try:
            # Encode the corrected text
            vec = encoder.encode(t)
            
            # Search in the store
            hits = store.search_traces(vec, top_k=1)
            
            if hits:
                trace_id, score = hits[0]
                
                # Retrieve the actual text content
                trace = store.get_trace(trace_id)
                if trace:
                    canonical = trace.content
                    
                    # Thresholds
                    AUTO_MERGE_THRESHOLD = 0.75
                    AMBIGUITY_THRESHOLD = 0.60
                    
                    if score > AUTO_MERGE_THRESHOLD:
                        if canonical != t:
                            logger.info(f"[FuzzyResolver] '{t}' -> '{canonical}' (similarity {score:.2f})")
                            return (canonical, trace_id)  # Return canonical + trace_id for fusion
                            
        except Exception as e:
            logger.warning(f"Fuzzy resolution failed: {e}")

    return (t, None)  # No canonicalization needed




class ConceptExtractor:
    def __init__(self, model_name: str = None):
        from .config import Config
        self.model_name = model_name or Config.embedding.GLINER_MODEL
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
