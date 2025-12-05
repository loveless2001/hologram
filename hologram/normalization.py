"""
Spelling and Normalization Pipeline.

This module implements a 4-stage pipeline to clean noisy input before it enters
the Gravity field. The goal is to prevent duplicate concepts, false mitosis,
and premature fusion.

Stages:
0. Tokenization Normalizer: Lightweight cleaning.
1. Spell-Correction: Dictionary-backed (SymSpell) correction.
2. Contextual Rewrite: LLM-based micro-correction (Optional).
3. Manifold Alignment: Semantic near-neighbor mapping via Gravity.
4. Concept Canonicalization: Deterministic formatting.
"""

import logging
import re
from typing import Optional, Callable, List, Any, Dict
import numpy as np

# Try importing SymSpell
try:
    from symspellpy import SymSpell, Verbosity
    import pkg_resources
    HAS_SYM_SPELL = True
except ImportError:
    HAS_SYM_SPELL = False

logger = logging.getLogger(__name__)

class NormalizationPipeline:
    def __init__(
        self, 
        gravity_field: Any = None, 
        llm_corrector_func: Optional[Callable[[str], str]] = None,
        enable_llm_correction: bool = False,
        encode_func: Optional[Callable[[str], np.ndarray]] = None
    ):
        """
        Initialize the normalization pipeline.
        
        Args:
            gravity_field: Reference to the GravityField (or Gravity) instance.
                           Used for whitelist checks and manifold alignment.
            llm_corrector_func: Callback function that takes a string and returns corrected string.
            enable_llm_correction: Whether to enable Stage 2 (LLM correction). Default False.
            encode_func: Function to encode text into a vector (compatible with gravity_field).
        """
        self.gravity = gravity_field
        self.llm_corrector = llm_corrector_func
        self.enable_llm = enable_llm_correction
        self.encode_func = encode_func
        
        # Initialize SymSpell
        self.sym_spell = None
        if HAS_SYM_SPELL:
            try:
                self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
                dictionary_path = pkg_resources.resource_filename(
                    "symspellpy", "frequency_dictionary_en_82_765.txt"
                )
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                logger.info("SymSpell dictionary loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load SymSpell dictionary: {e}")
                self.sym_spell = None

    def normalize(self, text: str) -> str:
        """
        Run the text through the 4-stage pipeline.
        
        Returns:
            The normalized text.
        """
        if not text:
            return ""
            
        # Stage 0: Tokenization Normalizer
        text = self._stage0_tokenize(text)
        
        # Stage 1: Spell-Correction (Dictionary)
        text = self._stage1_spell_correct(text)
        
        # Stage 2: Contextual Rewrite (LLM) - Optional
        if self.enable_llm and self.llm_corrector:
            text = self._stage2_llm_correct(text)
            
        # Stage 3: Manifold Alignment (Semantic)
        # We iterate over words to find near-miss concepts
        # This is expensive if we do it for every word with vector search.
        # Optimization: Only check words that are NOT in dictionary (if we had one) 
        # or words that SymSpell changed?
        # For now, let's only apply it to the *extracted concepts* via normalize_concept,
        # NOT the full sentence text, to avoid massive overhead and false positives on common words.
        # The user plan example "graviti" -> "gravity" implies it works on tokens.
        # But doing it on "The" -> "Thee" is bad.
        # Let's skip Stage 3 for full text normalization and only use it for concepts.
        # text = self._stage3_manifold_align(text) 
        
        # Stage 4: Canonicalization
        # This is mostly for concepts (lowercase, etc), but good for text too.
        text = self._stage4_canonicalize(text)
        
        return text

    def normalize_concept(self, concept: str) -> str:
        """
        Normalize a single extracted concept.
        This is where Stage 3 and 4 shine.
        """
        # Stage 3: Manifold Alignment
        concept = self._stage3_manifold_align(concept)
        
        # Stage 4: Canonicalization
        concept = self._stage4_canonicalize(concept)
        
        return concept

    # --- Stages ---

    def _stage0_tokenize(self, text: str) -> str:
        """Lightweight cleaning."""
        text = text.strip()
        text = text.replace("—", "-")
        text = text.replace("…", "...")
        text = text.replace("\u00a0", " ")  # non-breaking spaces
        
        # Canonicalize code tokens
        text = text.replace("::", " :: ")
        text = text.replace("->", " -> ")
        
        return " ".join(text.split())

    def _stage1_spell_correct(self, text: str) -> str:
        """Dictionary-backed spell correction."""
        if not self.sym_spell:
            return text
            
        out = []
        for word in text.split():
            # Check whitelist (Gravity concepts)
            if self._is_whitelisted(word):
                out.append(word)
                continue
                
            # SymSpell lookup
            suggestions = self.sym_spell.lookup(word, Verbosity.TOP, max_edit_distance=2)
            if suggestions:
                suggestion = suggestions[0].term
                # Only accept if confidence is high? SymSpell doesn't give probability easily without frequency.
                # But we trust top 1 for now.
                out.append(suggestion)
            else:
                out.append(word)
                
        return " ".join(out)

    def _stage2_llm_correct(self, text: str) -> str:
        """LLM-based contextual rewrite."""
        try:
            return self.llm_corrector(text)
        except Exception as e:
            logger.warning(f"LLM correction failed: {e}")
            return text

    def _stage3_manifold_align(self, text: str) -> str:
        """
        Semantic near-correction using Manifold/Gravity.
        Maps unknown tokens to known concepts if cosine > 0.65.
        """
        if not self.gravity or not self.encode_func:
            return text
            
        # Check if the text is already a known concept
        if self._is_whitelisted(text):
            return text
            
        # Embed the text
        try:
            vec = self.encode_func(text)
            
            # Find nearest neighbor in gravity field
            # Gravity.concepts is a dict, we need to search it.
            # If Gravity has an index (FAISS), use it.
            # GravityField (wrapper) has .search(), but Gravity (sim) might not.
            # Let's assume self.gravity is the GravityField wrapper or has .search
            
            hits = []
            if hasattr(self.gravity, 'search'):
                hits = self.gravity.search(vec, k=1)
            elif hasattr(self.gravity, 'sim') and hasattr(self.gravity.sim, 'concepts'):
                # Fallback: manual cosine scan (slow but works for prototype)
                # Or use the internal matrix if available
                # Let's try to use the most efficient method available
                # For now, let's assume we passed GravityField which has .search
                pass
            
            if hits:
                best_match, score = hits[0]
                if score > 0.65:
                    logger.info(f"[Manifold] '{text}' -> '{best_match}' ({score:.2f})")
                    return best_match
                    
        except Exception as e:
            logger.warning(f"Manifold alignment failed: {e}")
            
        return text

    def _stage4_canonicalize(self, text: str) -> str:
        """Deterministic mapping."""
        t = text.lower()
        t = t.replace("-", " ").replace("_", " ")
        return " ".join(t.split())

    def _is_whitelisted(self, word: str) -> bool:
        """Check if word exists in Gravity concepts."""
        if not self.gravity:
            return False
            
        # Handle Gravity wrapper (GravityField) vs raw Gravity
        if hasattr(self.gravity, 'sim') and hasattr(self.gravity.sim, 'concepts'):
            return word in self.gravity.sim.concepts
        
        # Fallback for raw Gravity instance
        if hasattr(self.gravity, 'concepts'):
            return word in self.gravity.concepts
            
        return False
