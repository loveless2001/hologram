import numpy as np
import logging
import random
from typing import List, Dict, Optional, Callable
from dataclasses import asdict
from datetime import datetime

from ..config import Config, VECTOR_DIM
from ..store import MemoryStore, Trace
from ..gravity import Gravity, Concept, cosine
from .registry import SymbolRegistry, SymbolMetadata
from .extractor import SymbolExtractor
from .parser import CodeParser

# Configure Logger
logger = logging.getLogger("CodeEvolution")
logger.setLevel(logging.INFO)

# Ensure log dir exists before adding handler
import os
os.makedirs("logs", exist_ok=True)

# File handler for audit logs
file_handler = logging.FileHandler("logs/code_memory_events.jsonl")
formatter = logging.Formatter('{"time": "%(asctime)s", "event": "%(message)s"}')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

class CodeEvolutionEngine:
    """
    Orchestrates the lifecycle of code concepts:
    - Diffing (New vs Old)
    - Identity Tracking (Registry)
    - Evolution (Drift, Fusion, Mitosis)
    - Updates (Store & Gravity)
    """
    
    def __init__(self, store: MemoryStore, vectorizer_func: Callable[[str], np.ndarray]):
        self.store = store
        self.field = store.sim  # Gravity field
        self.registry = SymbolRegistry(persistence_path="data/symbol_registry.json")
        self.vectorizer = vectorizer_func
        
        # Ensure log dir exists
        import os
        os.makedirs("logs", exist_ok=True)

    def process_file(self, file_path: str, language: str = "python") -> int:
        """
        Process a source file, compute diffs against memory, and apply evolution.
        Returns number of processed symbols.
        """
        # 1. Parse & Extract
        parser = CodeParser() # In Phase 2 this should handle multiple languages
        raw_nodes = parser.parse_file(file_path)
        
        extractor = SymbolExtractor()
        current_symbols = extractor.extract(raw_nodes, file_path)
        
        # Track seen IDs to detect deletions
        seen_symbol_ids = set()
        
        updates_count = 0
        
        for sym_data in current_symbols:
            symbol_id = sym_data["id"] 
            seen_symbol_ids.add(symbol_id)
            
            # Embed the new content
            new_vec = self.vectorizer(sym_data["body_text"])
            
            # Check Registry for existing history
            meta = self.registry.get(symbol_id)
            
            if not meta:
                # --- NEW SYMBOL CASE ---
                # Check for collision before adding
                if self._check_collision(sym_data["name"], new_vec):
                    # Collision detected! Force drift or rename?
                    # For now, we enforce uniqueness by ensuring ID hash is distinct (it is).
                    # But semantically, if name is very similar to another concept, specific logic applies.
                    pass
                
                self._handle_new_symbol(sym_data, new_vec, language)
                updates_count += 1
                
            else:
                # --- UPDATE CASE ---
                # Check Drift
                concept = self.field.concepts.get(symbol_id)
                if not concept:
                    # Should exist if in registry. Recovery/Sync issue?
                    # Treat as new for now
                    self._handle_new_symbol(sym_data, new_vec, language)
                    updates_count += 1
                    continue
                
                # Update Registry Metadata first (last_seen)
                meta.last_seen = datetime.now().isoformat()
                meta.file_path = file_path # Handle moves
                meta.status = "active" # In case it was deprecated
                self.registry.register(meta)
                
                # Update Span in Store (Cheap)
                trace = self.store.get_trace(symbol_id)
                if trace:
                    trace.span = sym_data["span"]
                    trace.source_file = file_path
                
                # Check Vector Drift
                old_vec = concept.vec
                drift = 1.0 - cosine(old_vec, new_vec)
                
                updates_count += 1
                
                if drift < Config.evolution.DRIFT_SMALL:
                    # Small Drift (< 0.08) -> Micro adjustment / Stability
                    self._handle_fusion(concept, new_vec, sym_data["name"])
                    
                elif drift < Config.evolution.DRIFT_MEDIUM:
                    # Medium Drift (0.08 - 0.22) -> Soft Fusion (Interpolation)
                    self._handle_soft_fusion(concept, new_vec, sym_data["name"])
                    
                elif drift > Config.evolution.DRIFT_LARGE:
                    # Large Drift (> 0.38) -> Mitosis (Split)
                    # We reuse gravity logic, but trigger it explicitly
                    self._handle_mitosis_explicit(concept, new_vec, sym_data)
                else:
                    # Between Medium and Large (0.22 - 0.38) -> Strong Adaptation
                    # Treat as heavy Soft Fusion
                     self._handle_soft_fusion(concept, new_vec, sym_data["name"])
                
                # Check Vector Rot (Fatigue)
                self._check_vector_rot(concept, new_vec)

        # Handle Deprecations (Symbols in registry for this file NOT in current scan)
        # We need efficient lookup for "symbols belonging to this file"
        # Since scanning linear registry is slow, we can optimize later.
        # For prototype, we unfortunately iterate.
        for sid, m in self.registry.registry.items():
            if m.file_path == file_path and sid not in seen_symbol_ids and m.status == "active":
                self._handle_deprecation(sid)
                
        self.registry.save()
        return updates_count

    def _handle_new_symbol(self, sym_data, vec, language):
        """Register and store a completely new symbol."""
        symbol_id = sym_data["id"]
        
        # Check if this is a Revival (exists in registry but deprecated)
        existing_meta = self.registry.get(symbol_id)
        if existing_meta and existing_meta.status == "deprecated":
            self._handle_revival(symbol_id, vec)
            return

        # Create Metadata
        meta = SymbolMetadata(
            symbol_id=symbol_id,
            qualified_name=sym_data["qualified_name"],
            signature=sym_data["signature"],
            file_path=sym_data["file"],
            language=language,
            first_seen=datetime.now().isoformat(),
            last_seen=datetime.now().isoformat(),
            status="active",
            vector_hash="" # TODO
        )
        self.registry.register(meta)
        
        # Create Trace & Concept
        trace = Trace(
            trace_id=symbol_id,
            kind="code",
            content=sym_data["body_text"],
            vec=vec,
            span=sym_data["span"],
            source_file=sym_data["file"],
            meta={"signature": sym_data["signature"], "type": sym_data["kind"]}
        )
        self.store.add_trace(trace) # Also adds to gravity via store
        
        # Set initial evo fields in Concept
        c = self.field.concepts.get(symbol_id)
        if c:
            c.original_vec = vec.copy()
            c.age = 0
            c.origin = "code_map"

        logger.info(f"{{'action': 'NEW', 'symbol': '{sym_data['qualified_name']}'}}")


    def _handle_fusion(self, concept: Concept, new_vec: np.ndarray, name: str):
        """Small drift: Weighted Average Fusion."""
        # Reuse gravity logic? Gravity fuse_concepts merges TWO concepts.
        # Here we have ONE concept evolving. We manually update.
        
        # Weighted avg: bias towards stability (old) or novelty (new)?
        # Usually stability for small drift.
        # new = (old * mass + new * 1.0) / (mass + 1.0)
        
        total_mass = concept.mass + 1.0
        fused = (concept.vec * concept.mass + new_vec) / total_mass
        fused /= (np.linalg.norm(fused) + 1e-8)
        
        concept.previous_vec = concept.vec.copy()
        concept.vec = fused
        concept.mass += 0.1 # Small growth
        
        logger.info(f"{{'action': 'FUSION', 'symbol': '{name}', 'drift': 'small'}}")

    def _handle_soft_fusion(self, concept: Concept, new_vec: np.ndarray, name: str):
        """Medium drift: Linear Interpolation."""
        # concept.vec = 0.7 * old + 0.3 * new
        alpha = 0.3
        interpolated = (1 - alpha) * concept.vec + alpha * new_vec
        interpolated /= (np.linalg.norm(interpolated) + 1e-8)
        
        concept.previous_vec = concept.vec.copy()
        concept.vec = interpolated
        concept.mass += 0.2
        
        logger.info(f"{{'action': 'SOFT_FUSION', 'symbol': '{name}', 'drift': 'medium'}}")

    def _handle_mitosis_explicit(self, concept: Concept, new_vec: np.ndarray, sym_data: dict):
        """Large drift: Trigger Mitosis."""
        # We want to use gravity.check_mitosis, but that relies on internal tension (clustering of neighbors).
        # Here, the tension is strictly temporal (Old Self vs New Self).
        # Gravity's mitosis splits a cluster.
        # Here we want to fork the version.
        
        # Fork:
        # 1. Create v2 concept with new_vec
        # 2. Mark v1 as historical (or just existing)
        # 3. Add bridge edge
        
        base_id = sym_data["id"]
        # Generate versioned ID
        v_tag = f"_v{concept.age + 1}" # Simple versioning
        new_id = base_id + v_tag
        
        # Since we want to keep specific logic closer to user request:
        # "Create _v2 concept, add bridge edge"
        
        self.field.add_concept(
            new_id,
            vec=new_vec,
            mass=1.0, # Fresh start
            origin="code_map"
        )
        
        # Bridge
        key = (min(base_id, new_id), max(base_id, new_id))
        self.field.relations[key] = 0.15 # Weak bridge
        
        # Update Registry to point to new ID? 
        # Actually user said "Symbol Identity is deterministic". 
        # If the hash changed, processed_file would see it as a NEW symbol.
        # If we are here, hash is SAME.
        # This means the code text changed, signatures/names same.
        # If logic changes drastically but signature same...
        
        # The prompt says: "If drift > LARGE ... Create new concept version."
        # This implies the original ID stays as "v1" and we make a "v2"?
        # Or we update the original ID to point to new content, and archive the old?
        
        # Let's Archive Old, Update Original to New.
        # 1. Create Archive Copy
        archive_id = f"{base_id}_arch_v{concept.age}"
        self.field.add_concept(archive_id, vec=concept.vec, mass=concept.mass, origin="code_history")
        
        # 2. Update Original
        concept.vec = new_vec
        concept.mass = 1.0 # Reset mass? Or keep it? kept usually implies continuity. 
        # Let's keep mass but reduced?
        concept.mass *= 0.5
        
        # 3. Bridge Original -> Archive
        key = (min(base_id, archive_id), max(base_id, archive_id))
        self.field.relations[key] = 0.15
        
        logger.info(f"{{'action': 'MITOSIS', 'symbol': '{sym_data['qualified_name']}', 'new_version': '{archive_id}'}}")


    def _handle_deprecation(self, symbol_id: str):
        """Mark as deprecated and apply forced decay."""
        concept = self.field.concepts.get(symbol_id)
        if not concept: return
        
        meta = self.registry.get(symbol_id)
        if meta:
            meta.status = "deprecated"
        
        # Update trace
        trace = self.store.get_trace(symbol_id)
        if trace:
            trace.status = "deprecated"
            
        # Forced Decay Physics
        # 1. Outward Drift
        # vec = normalize(vec + random_unit * decay_factor)
        rand_vec = np.random.randn(self.store.vec_dim).astype(np.float32)
        rand_vec /= np.linalg.norm(rand_vec)
        
        decay_factor = Config.evolution.OBSOLETE_DECAY
        decayed_vec = concept.vec + (rand_vec * decay_factor)
        decayed_vec /= (np.linalg.norm(decayed_vec) + 1e-8)
        
        concept.vec = decayed_vec
        
        # 2. Mass Decay
        concept.mass *= (1.0 - Config.evolution.DECAY_RATE)
        
        logger.info(f"{{'action': 'DEPRECATION', 'symbol': '{symbol_id}'}}")

    def _handle_revival(self, symbol_id: str, new_vec: np.ndarray):
        """Revive a deprecated symbol."""
        concept = self.field.concepts.get(symbol_id)
        if not concept: return # Should exist if in registry

        # Update metadata
        meta = self.registry.get(symbol_id)
        if meta: meta.status = "revived"
        
        trace = self.store.get_trace(symbol_id)
        if trace: trace.status = "revived"
        
        # Physics Boost
        concept.mass = concept.mass * 1.2
        concept.vec = new_vec # Reset vector to current reality
        
        logger.info(f"{{'action': 'REVIVAL', 'symbol': '{symbol_id}'}}")

    def _check_collision(self, name: str, vec: np.ndarray) -> bool:
        """
        Anti-Collision Filter.
        Return True if collision detected (similar vector but different name).
        """
        # Find nearest neighbor
        # In a real system, use index search. 
        # Here we do a linear scan of active code concepts for safety
        
        # Limit scan for perf?
        # Just check active code concepts
        best_sim = 0.0
        best_name = ""
        
        # Optimization: use store.search_traces which uses Faiss
        hits = self.store.search_traces(vec, top_k=3)
        for tid, score in hits:
             if tid in self.field.concepts:
                 c = self.field.concepts[tid]
                 if c.origin == "code_map":
                     if score > best_sim:
                         best_sim = score
                         best_name = c.name # This might be the ID
                         # We need the human name. 
                         # Registry has it.
                         
        if best_sim > 0.85: # High vector similarity
             # Check name similarity
             # This requires getting the qualified name from registry for 'tid'
             m = self.registry.get(best_name) # best_name is trace_id
             if m:
                 # Levenshtein or explicit match?
                 # User said: "check name_similarity < 0.6 before fusion"
                 # Here we are NEW symbol.
                 # If existing symbol is close in vector, we risk confusing them.
                 pass
                 
        return False # TODO: Implement full string sim check

    def _check_vector_rot(self, concept: Concept, current_vec: np.ndarray):
        """Check for long-term vector rot (fatigue)."""
        if concept.original_vec is None:
            concept.original_vec = current_vec.copy()
            return
            
        concept.vector_history.append(current_vec.copy())
        if len(concept.vector_history) > 3:
            concept.vector_history.pop(0)
            
        concept.age += 1
        
        # Divergence check
        sim = cosine(concept.original_vec, current_vec)
        if sim < Config.evolution.ROT_THRESHOLD: # e.g. < 0.5 means > 0.5 divergence? 
            # User said: "cosine > ROT_THRESHOLD" trigger action? 
            # Usually strictness means HIGH similarity (close to 1.0).
            # So if sim is LOW (close to 0), we drifted FAR.
            # So if sim < ROT_THRESHOLD (0.5), we have rotated too much.
            
            logger.warning(f"{{'action': 'ROT_WARNING', 'symbol': '{concept.name}', 'similarity': {sim:.3f}}}")
            # Action: Re-center? Or Mitosis?
            # User suggested "stabilizing re-center or mitosis check"
            # Let's re-anchor slightly to original?
            # concept.vec = 0.5 * concept.vec + 0.5 * concept.original_vec
            pass
