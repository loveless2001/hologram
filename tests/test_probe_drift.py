

import numpy as np
import sys
import os

# Ensure we can import hologram
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.gravity import Gravity, Probe, TIER_DOMAIN
from hologram.api import Hologram

def test_gravity_probe_physics():
    """Test pure physics drift in Gravity class."""
    gravity = Gravity(dim=2)
    
    # Create two clusters
    # Cluster A: centered at [1, 1]
    gravity.add_concept("A1", vec=np.array([1.0, 1.0]), mass=10.0) # High mass
    gravity.add_concept("A2", vec=np.array([1.1, 0.9]), mass=1.0)
    
    # Cluster B: centered at [-1, -1]
    gravity.add_concept("B1", vec=np.array([-1.0, -1.0]), mass=10.0)
    gravity.add_concept("B2", vec=np.array([-0.9, -1.1]), mass=1.0)
    
    # Spawn probe nearer to A, but slightly offset
    # Start at [0.5, 0.5] -> Should drift to A
    start_vec = np.array([0.5, 0.5])
    
    probe = gravity.run_probe(start_vec, max_steps=10)
    
    print(f"Probe History: {len(probe.history)} steps")
    for i, step in enumerate(probe.history):
        print(f"Step {i}: pos={step.position}")
        
    final_pos = probe.vec
    
    # Check if closer to A ([0.707, 0.707] normalized) than B
    dist_A = np.linalg.norm(final_pos - np.array([0.707, 0.707]))
    dist_B = np.linalg.norm(final_pos - np.array([-0.707, -0.707]))
    
    assert dist_A < dist_B, "Probe should drift towards Cluster A"
    assert len(probe.history) > 0
    
    # Check history structure
    step0 = probe.history[0]
    assert step0.neighbors, "Should have neighbors"
    assert step0.chosen, "Should have used some neighbors"

def test_hologram_integration():
    """Test full API integration."""
    holo = Hologram.init(use_gravity=True, encoder_mode="hash")
    
    # Ingest some concepts
    holo.add_text("glyph:science", "Physics and Chemistry")
    holo.add_text("doc:1", "Newtonian mechanics deals with forces and mass.")
    holo.add_text("doc:2", "Quantum mechanics is about wave functions.")
    holo.add_text("glyph:art", "Painting and Sculpture")
    holo.add_text("doc:3", "Impressionism captures light.")
    
    # Search with drift
    query = "force and mass"
    result = holo.search_with_drift(query, top_k_traces=5, probe_steps=5)
    
    assert result["probe"] is not None
    assert result["tree"] is not None
    assert result["results"] is not None
    
    print("Tree Root:", result["tree"].root_id)
    print("Nodes found:", result["tree"].nodes.keys())
    
    # Check if doc:1 is in results (should be high score)
    trace_ids = [r["trace"].trace_id for r in result["results"]]
    # doc:1 ID might be hash-based, let's check content or if we can find it
    
    # We added it with explicit ID "doc:1"? No, add_text trace_id defaults to hash unless specified.
    # Wait, add_text signature: def add_text(self, glyph_id: str, text: str, trace_id: Optional[str] = None...
    # I passed "doc:1" as text? No, wait.
    # holo.add_text("glyph:science", "Physics...")
    # I want to verify specific traces.
    
    # Let's clean up and do explicit IDs
    trace_id_1 = holo.add_text("glyph:science", "Newtonian mechanics forces mass", trace_id="trace:newton")
    
    result = holo.search_with_drift("force mass", top_k_traces=5)
    
    found = False
    for res in result["results"]:
        if res["trace"].trace_id == "trace:newton":
            found = True
            break
            
    assert found, "Should find the Newtonian trace via drift"

if __name__ == "__main__":
    # Manually run if executed as script
    try:
        test_gravity_probe_physics()
        print("Physics Test Passed")
        test_hologram_integration()
        print("Integration Test Passed")
    except AssertionError as e:
        print(f"Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)
