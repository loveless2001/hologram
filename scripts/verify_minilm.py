
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def verify_minilm():
    print("Initializing Hologram with encoder_mode='minilm'...")
    try:
        holo = Hologram.init(encoder_mode="minilm", use_gravity=False)
    except Exception as e:
        print(f"Failed to init: {e}")
        return

    print(f"Encoder initialized. Dim: {holo.store.vec_dim}")
    
    # Test semantic similarity
    pairs = [
        ("gravity", "physics"),
        ("gravity", "banana"),
        ("king", "queen"),
        ("king", "apple")
    ]
    
    print("\nSimilarity checks:")
    for w1, w2 in pairs:
        v1 = holo.text_encoder.encode(w1)
        v2 = holo.text_encoder.encode(w2)
        sim = cosine(v1, v2)
        print(f"Sim({w1}, {w2}) = {sim:.4f}")
        
    # Check if gravity-physics > gravity-banana
    v_grav = holo.text_encoder.encode("gravity")
    v_phys = holo.text_encoder.encode("physics")
    v_ban = holo.text_encoder.encode("banana")
    
    if cosine(v_grav, v_phys) > cosine(v_grav, v_ban):
        print("\n[PASS] Semantic check passed: gravity is closer to physics than banana.")
    else:
        print("\n[FAIL] Semantic check failed.")

if __name__ == "__main__":
    verify_minilm()
