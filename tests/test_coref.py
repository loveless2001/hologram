import pytest
import numpy as np
from hologram.coref import resolve, HAS_FASTCOREF
from hologram.gravity import Gravity
from hologram.api import Hologram
from hologram.config import Config

# Mock fastcoref if not installed to allow tests to run (partially)
if not HAS_FASTCOREF:
    pytest.skip("fastcoref not installed", allow_module_level=True)

def test_basic_resolution():
    text = "The engine failed. It was overheating."
    resolved, coref_map = resolve(text)
    
    # We expect "It" to be resolved to "The engine"
    # Note: Exact string match depends on model output
    assert "It" in coref_map or "it" in coref_map
    assert "engine" in coref_map.get("It", "").lower() or "engine" in coref_map.get("it", "").lower()
    
    # Check resolved text
    assert "engine" in resolved
    assert "It" not in resolved or "The engine" in resolved

def test_gravity_fallback():
    # Setup a small gravity field
    g = Gravity(dim=4)
    
    # Add concepts
    # "fusion" vector
    v_fusion = np.array([1.0, 0.0, 0.0, 0.0])
    g.add_concept("fusion", vec=v_fusion, mass=10.0, tier=1)
    
    # "drift" vector (orthogonal)
    v_drift = np.array([0.0, 1.0, 0.0, 0.0])
    g.add_concept("drift", vec=v_drift, mass=5.0, tier=1)
    
    # Test sentence: "This is powerful."
    # If we assume "This" refers to "fusion" (closest vector)
    # We need to mock the encoder to return something close to fusion
    
    # Mock encode method
    original_encode = g.encode
    g.encode = lambda x: np.array([0.9, 0.1, 0.0, 0.0]) # Close to fusion
    
    resolved = g.resolve_pronoun("This is powerful.", "This")
    assert resolved == "fusion"
    
    # Restore
    g.encode = original_encode

def test_integration_add_text(isolated_hologram):
    # Initialize Hologram with coref enabled
    Config.coref.ENABLE_COREF = True
    Config.coref.ENABLE_GRAVITY_FALLBACK = True
    
    h = isolated_hologram
    
    # Add a concept manually to the field so fallback has something to find
    # (Hash encoder is deterministic)
    h.field.add("engine", vec=h.text_encoder.encode("engine"), tier=1)
    
    # Add text with pronoun
    text = "The engine is loud. It vibrates."
    trace_id = h.add_text("test_glyph", text)
    
    trace = h.store.get_trace(trace_id)
    
    assert trace is not None
    assert trace.resolved_text is not None
    assert trace.coref_map is not None
    
    # "It" should be resolved either by fastcoref or fallback
    # Since we use hash encoder, fallback might be random if fastcoref fails or is not confident.
    # But fastcoref should handle this simple case.
    
    if HAS_FASTCOREF:
        # Check if map is populated
        # Note: "It" might map to "The engine"
        found = False
        for k, v in trace.coref_map.items():
            if "it" in k.lower() and "engine" in v.lower():
                found = True
                break
        assert found, f"Coref map failed: {trace.coref_map}"

def test_no_false_fusions(isolated_hologram):
    # Ensure that resolved pronouns do NOT trigger concept fusion incorrectly
    # i.e. "It" resolving to "engine" shouldn't make "It" a concept alias of "engine" globally
    
    Config.coref.ENABLE_COREF = True
    h = isolated_hologram
    
    h.add_text("g1", "The engine is loud. It vibrates.")
    
    # Check concepts in field
    # We should NOT see a concept named "It" fused into "engine"
    # The resolution happens at text level, not concept level (unless gravity fallback does it?)
    
    # Gravity fallback returns a concept name, but it doesn't create a NEW concept "It".
    # It just maps the word "It" to the existing concept "engine".
    
    assert "It" not in h.field.sim.concepts
    assert "it" not in h.field.sim.concepts
