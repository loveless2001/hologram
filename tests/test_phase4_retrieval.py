
# tests/test_phase4_retrieval.py
import numpy as np
import sys
import os
import pytest

# Ensure we can import hologram
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram
from hologram.gravity import Gravity, Probe, cosine
from hologram.retrieval import extract_local_field
from hologram.smi import MemoryPacket

def test_probe_physics():
    print("\n--- Testing Probe Physics ---")
    g = Gravity(dim=64, seed=42)
    
    # Create a massive concept
    target_vec = np.random.rand(64).astype('float32')
    target_vec /= np.linalg.norm(target_vec)
    g.add_concept("target", vec=target_vec, mass=10.0, tier=1) # Heavy!
    
    # Spawn probe nearby
    start_vec = target_vec + np.random.normal(0, 0.1, 64).astype('float32')
    start_vec /= np.linalg.norm(start_vec)
    
    # NEW API: Probe init
    probe = Probe(name="test_probe", vec=start_vec)
    
    # Check initial distance
    dist_start = cosine(probe.vec, target_vec)
    print(f"Start Similarity: {dist_start:.3f}")
    
    # Step simulation (NEW API: g.step_probe)
    g.step_probe(probe, alpha=0.1)  # Small step
    
    # Check if moved closer
    dist_end = cosine(probe.vec, target_vec)
    print(f"End Similarity:   {dist_end:.3f}")
    
    assert dist_end > dist_start, "Probe should drift towards massive concept"
    assert len(probe.history) >= 1, "History should record steps"
    print("✓ Probe drift verified")

def test_retrieval_packet_structure():
    print("\n--- Testing Retrieval Packet Structure ---")
    holo = Hologram.init(use_clip=False)
    
    # Add some data
    holo.add_text("g1", "The quick brown fox jumps over the lazy dog.")
    holo.add_text("g1", "Foxes are clever animals.")
    
    # Retrieve
    packet = holo.retrieve("brown fox")
    
    assert isinstance(packet, MemoryPacket)
    assert packet.seed == "brown fox"
    assert packet.trajectory_steps >= 0
    
    print("Nodes found:", [n['name'] for n in packet.nodes])
    
    # Check JSON serialization
    json_str = packet.to_json()
    assert "seed" in json_str
    assert "nodes" in json_str
    print("✓ Packet structure verified")

def test_smi_prompt_generation():
    print("\n--- Testing SMI Prompt Generation ---")
    packet = MemoryPacket(
        seed="test",
        nodes=[{"name": "concept_a", "mass": 1.5, "age": 10}],
        edges=[{"a": "concept_a", "b": "concept_b", "relation": 0.9, "tension": 0.1}],
        glyphs=[{"id": "glyph_1", "mass": 2.0, "similarity": 0.8}],
        trajectory_steps=5
    )
    
    block = packet.to_prompt_block()
    
    assert "concept_a" in block
    assert "mass=1.5" in block
    assert "glyph_1" in block
    assert "tension=0.1" in block
    print("✓ SMI prompt generation verified")

if __name__ == "__main__":
    try:
        test_probe_physics()
        test_retrieval_packet_structure()
        test_smi_prompt_generation()
        print("\nAll Phase 4 Retrieval tests passed!")
    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
