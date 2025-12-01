
import time
import sys
import os
import json
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram

def profile_search(store_path: str):
    print(f"Profiling search on {store_path}...")
    
    if not os.path.exists(store_path):
        print("Store not found. Run profile_ingest.py first.")
        return

    # Load store
    start_load = time.time()
    holo = Hologram.load(store_path, use_clip=False, use_gravity=True)
    load_time = time.time() - start_load
    print(f"Load time: {load_time:.4f}s")
    
    queries = ["gravity", "quantum physics", "black hole", "relativity", "energy"]
    
    # Profile search_text
    latencies = []
    for q in queries:
        start = time.time()
        holo.search_text(q, top_k=10)
        latencies.append(time.time() - start)
        
    avg_search = sum(latencies) / len(latencies)
    print(f"Avg search_text latency (top-10): {avg_search*1000:.2f}ms")
    
    # Profile project2d (viz)
    start_viz = time.time()
    holo.field_state()
    viz_time = time.time() - start_viz
    print(f"project2d latency: {viz_time*1000:.2f}ms")
    
    return {
        "load_time": load_time,
        "search_ms_top10": avg_search * 1000,
        "viz_ms": viz_time * 1000
    }

if __name__ == "__main__":
    results = profile_search("data/perf_store.json")
    if results:
        # Save combined results if needed, or just print
        pass
