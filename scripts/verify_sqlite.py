
import sys
import os
import shutil
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram

def verify_sqlite():
    db_path = "data/test_sqlite.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        
    print(f"Testing SQLite backend at {db_path}...")
    
    # 1. Init and populate
    holo = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    print("Adding traces...")
    holo.add_text("g1", "Hello world", trace_id="t1")
    holo.add_text("g1", "Another trace", trace_id="t2")
    holo.add_text("g2", "Gravity is cool", trace_id="t3")
    
    print(f"Saving to {db_path}...")
    holo.save(db_path)
    
    # 2. Load back
    print("Loading back...")
    holo2 = Hologram.load(db_path, encoder_mode="minilm", use_gravity=True)
    
    # 3. Verify
    print(f"Loaded {len(holo2.store.traces)} traces.")
    print(f"Loaded {len(holo2.store.glyphs)} glyphs.")
    print(f"Gravity concepts: {len(holo2.field.sim.concepts)}")
    
    assert len(holo2.store.traces) == 3
    assert len(holo2.store.glyphs) == 2
    assert "t1" in holo2.store.traces
    assert holo2.store.traces["t1"].content == "Hello world"
    
    # Verify gravity state restored
    # "Hello world" -> "hello", "world" concepts
    # "Another trace" -> "another", "trace"
    # "Gravity is cool" -> "gravity", "cool"
    # Total concepts ~6
    if len(holo2.field.sim.concepts) > 0:
        print("[PASS] Gravity state restored.")
    else:
        print("[FAIL] Gravity state empty.")

    print("[PASS] SQLite verification complete.")

if __name__ == "__main__":
    verify_sqlite()
