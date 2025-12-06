import pytest
import numpy as np
import threading
import time
from hologram.store import MemoryStore, Trace
from hologram.gravity import Gravity, Concept, cosine
from hologram.config import Config
from hologram.code_map.evolution import CodeEvolutionEngine
from hologram.code_map.registry import SymbolRegistry, SymbolMetadata

# --- MOCKS ---
class MockGravity(Gravity):
    def check_mitosis(self, name, **kwargs):
        # Allow spying on calls
        if hasattr(self, 'mitosis_calls'):
            self.mitosis_calls.append(name)
        return super().check_mitosis(name, **kwargs)

class MockStore(MemoryStore):
    def __init__(self):
        super().__init__(vec_dim=384)
        self.sim = MockGravity(dim=384)
        pass # Override heavier init if needed

def mock_vectorizer(text: str) -> np.ndarray:
    # Deterministic "vector" based on text length + ascii sum
    # This allows us to control "drift" precisely by changing text
    np.random.seed(len(text))
    # Base vector
    vec = np.random.randn(384).astype(np.float32)
    # perturbation based on content
    val = sum(ord(c) for c in text) % 100
    vec += (val * 0.01) # Small perturbation
    return vec / np.linalg.norm(vec)

# Wrapper to control drift
def make_vector(seed_val: int = 0, base_vec: np.ndarray = None, noise: float = 0.0) -> np.ndarray:
    if base_vec is not None:
        rng = np.random.RandomState(seed_val)
        noise_vec = rng.randn(384).astype(np.float32)
        noise_vec /= np.linalg.norm(noise_vec)
        new_vec = base_vec + (noise_vec * noise)
        return new_vec / np.linalg.norm(new_vec)
    else:
        rng = np.random.RandomState(seed_val)
        vec = rng.randn(384).astype(np.float32)
        return vec / np.linalg.norm(vec)

# --- TESTS ---

@pytest.fixture
def evolution_setup(tmp_path):
    store = MockStore()
    
    # Mock vectorizer to be controllable
    # We will patch it inside individual tests if needed
    engine = CodeEvolutionEngine(store, mock_vectorizer)
    
    # Use tmp path for registry
    engine.registry = SymbolRegistry(persistence_path=str(tmp_path / "registry.json"))
    
    return engine, store

def test_new_symbol_ingestion(evolution_setup):
    engine, store = evolution_setup
    
    # Create a dummy python file
    code = """
def test_func():
    print("Hello")
"""
    import os
    with open("test_new.py", "w") as f:
        f.write(code)
        
    count = engine.process_file("test_new.py")
    assert count == 1
    
    # Verify Trace
    # We need to know the ID. Extractor uses hash.
    # scan store traces
    assert len(store.traces) == 1
    trace = list(store.traces.values())[0]
    assert trace.status == "active"
    assert "test_func" in trace.meta["signature"]
    
    # Verify Registry
    assert len(engine.registry.registry) == 1
    meta = list(engine.registry.registry.values())[0]
    assert meta.qualified_name == "test_func"
    
    os.remove("test_new.py")

def test_fusion_small_drift(evolution_setup):
    engine, store = evolution_setup
    
    # 1. Setup existing concept
    concept_id = "test_id"
    vec_v1 = make_vector(seed_val=0) # Base
    
    c = Concept(name=concept_id, vec=vec_v1.copy(), mass=1.0)
    c.origin = "code_map"
    c.original_vec = vec_v1.copy()
    store.sim.concepts[concept_id] = c
    
    trace = Trace(trace_id=concept_id, kind="code", content="old", vec=vec_v1)
    store.traces[concept_id] = trace
    
    # Register it
    meta = SymbolMetadata(
        symbol_id=concept_id, qualified_name="func", signature="func()", 
        file_path="t.py", language="python", first_seen="", last_seen="", status="active", vector_hash=""
    )
    engine.registry.register(meta)

    # 2. Trigger Update with Small Drift
    # vec_v2 close to v1 (noise 0.1 gives roughly 0.005 drift in Cosine? Need to tune)
    # Cosine distance ~ noise^2 / 2 for small noise
    # 0.1 -> 0.005 drift. 0.3 -> 0.045 drift. 0.4 -> 0.08 drift.
    vec_v2 = make_vector(seed_val=1, base_vec=vec_v1, noise=0.3)
    drift = 1.0 - cosine(vec_v1, vec_v2) 
    # Ensure drift is SMALL (<0.08)
    assert drift < 0.08 and drift > 0.001
    
    # We need to inject this into process_file, but parsing real file is hard to align with mock vectors.
    # Let's call _handle_fusion directly to verify logic
    engine._handle_fusion(c, vec_v2, "func")
    
    # Verify Fusion (Weighted Average)
    # Mass was 1.0, grew by 0.1
    assert c.mass == 1.1
    # Vector should be blended
    # (v1*1 + v2*1)/2 approx
    assert not np.array_equal(c.vec, vec_v1)
    assert not np.array_equal(c.vec, vec_v2)
    # Should be close to both
    assert 1.0 - cosine(c.vec, vec_v2) < 0.05

def test_mitosis_large_drift(evolution_setup):
    engine, store = evolution_setup
    store.sim.mitosis_calls = []
    
    # 1. Setup
    concept_id = "test_mitosis"
    vec_v1 = make_vector(seed_val=0)
    c = Concept(name=concept_id, vec=vec_v1.copy(), mass=5.0) # High mass
    c.origin = "code_map"
    c.age = 5
    store.sim.concepts[concept_id] = c
    
    # Register
    meta = SymbolMetadata(concept_id, "func", "func()", "t.py", "python", "", "", "active", "")
    engine.registry.register(meta)
    
    # 2. Update with Large Drift
    vec_v2 = make_vector(seed_val=1) # Totally diff seed -> Orthogonal
    drift = 1.0 - cosine(vec_v1, vec_v2)
    assert drift > 0.4
    
    # Mock data packet
    sym_data = {"id": concept_id, "qualified_name": "func"}
    
    engine._handle_mitosis_explicit(c, vec_v2, sym_data)
    
    # Verify Logic
    # 1. Archive created
    archive_id = f"{concept_id}_arch_v{c.age}"
    assert archive_id in store.sim.concepts
    
    # 2. Original Updated
    assert np.allclose(c.vec, vec_v2, atol=1e-5)
    
    # 3. Bridge exists
    key = (min(concept_id, archive_id), max(concept_id, archive_id))
    assert key in store.sim.relations

def test_deprecation_and_revival(evolution_setup):
    engine, store = evolution_setup
    
    # 1. Setup
    sid = "dep_test"
    vec = make_vector(seed_val=0)
    c = Concept(name=sid, vec=vec.copy(), mass=1.0)
    store.sim.concepts[sid] = c
    store.traces[sid] = Trace(sid, "code", "content", vec)
    
    meta = SymbolMetadata(sid, "func", "func()", "t.py", "python", "", "", "active", "")
    engine.registry.register(meta)
    
    # 2. Deprecate
    engine._handle_deprecation(sid)
    
    assert engine.registry.get(sid).status == "deprecated"
    assert store.traces[sid].status == "deprecated"
    # Mass decay
    assert c.mass < 1.0
    # Vector drift (randomized, just check it changed)
    assert not np.allclose(c.vec, vec)
    
    # 3. Revive
    new_vec = make_vector(seed_val=1)
    engine._handle_revival(sid, new_vec)
    
    assert engine.registry.get(sid).status == "revived"
    assert c.mass > 1.0 # Boosted
    assert np.allclose(c.vec, new_vec)

if __name__ == "__main__":
    # Allow running directly
    pytest.main([__file__])
