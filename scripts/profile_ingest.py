
import time
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram

def profile_ingest(kb_path: str, store_path: str):
    print(f"Profiling ingestion of {kb_path}...")
    
    # Initialize Hologram (fresh start)
    if os.path.exists(store_path):
        os.remove(store_path)
        
    holo = Hologram.init(use_clip=False, use_gravity=True)
    
    with open(kb_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    start_time = time.time()
    
    for i, line in enumerate(lines):
        holo.add_text(glyph_id="ingest_test", text=line, do_extract_concepts=True)
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(lines)} lines...", end="\r")
            
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nIngestion complete.")
    print(f"Total lines: {len(lines)}")
    print(f"Total time: {duration:.4f}s")
    print(f"Avg time per line: {duration/len(lines):.4f}s")
    
    # Save for search profiling
    holo.save(store_path)
    print(f"Store saved to {store_path}")
    
    return {
        "num_lines": len(lines),
        "total_time": duration,
        "avg_time_per_line": duration/len(lines)
    }

if __name__ == "__main__":
    kb_file = "data/kbs/physics_biased_kb.txt"
    if len(sys.argv) > 1:
        kb_file = sys.argv[1]
        
    if not os.path.exists(kb_file):
        # Create a dummy KB if not exists
        print(f"KB file {kb_file} not found. Creating dummy...")
        os.makedirs(os.path.dirname(kb_file), exist_ok=True)
        with open(kb_file, "w") as f:
            for i in range(100):
                f.write(f"This is a test sentence number {i} about gravity and physics.\n")
    
    profile_ingest(kb_file, "data/perf_store.json")
