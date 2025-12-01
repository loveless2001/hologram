
import time
import sys
import os
import concurrent.futures
from typing import List
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.api import Hologram
from hologram.text_utils import extract_concepts

# Mock GLiNER if not available or just use extract_concepts
# extract_concepts currently uses GLiNER if installed, or regex fallback?
# Let's assume extract_concepts is the function we want to parallelize.

def process_chunk(lines: List[str]):
    """
    Extract concepts from a batch of lines.
    Run in ThreadPool (GLiNER releases GIL? or IO bound?)
    Actually, if GLiNER is CPU heavy, ProcessPool is better.
    But let's stick to the plan: ThreadPool for extraction (maybe it calls an API or is fast enough?)
    """
    results = []
    for line in lines:
        concepts = extract_concepts(line)
        results.append((line, concepts))
    return results

def build_kb_concurrent(kb_path: str, store_path: str, batch_size: int = 32):
    print(f"Building KB from {kb_path} concurrently...")
    
    # Init Hologram (Main Process)
    # We need the encoder in the main process to add to store?
    # Or we can compute embeddings in parallel?
    # TextMiniLM is not pickle-able easily across processes if it holds a model?
    # Actually, it's better to run the model in the main process with batching, 
    # OR spawn workers that load their own model.
    # Spawning workers with models is heavy on RAM.
    # Best approach: 
    # 1. ThreadPool for text extraction (if IO/API) or ProcessPool (if CPU).
    # 2. Main thread does batch embedding (GPU/CPU vectorized).
    
    holo = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    with open(kb_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
        
    print(f"Total lines: {len(lines)}")
    
    start_time = time.time()
    
    # Chunk lines
    chunks = [lines[i:i + batch_size] for i in range(0, len(lines), batch_size)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all extraction jobs
        future_to_chunk = {executor.submit(process_chunk, chunk): chunk for chunk in chunks}
        
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk_results = future.result()
            
            # Now we have [(line, [concepts]), ...]
            # We need to embed and add to store.
            # Doing this in main thread allows batching embeddings if the encoder supports it.
            # TextMiniLM.encode is single item currently. 
            # We should add encode_batch to TextMiniLM.
            
            for line, concepts in chunk_results:
                # Add trace (embeds line)
                # This calls holo.add_text -> manifold.align_text -> encoder.encode
                # If we want speed, we should batch encode.
                # But holo.add_text is one by one.
                # For now, just call add_text.
                
                holo.add_text("ingest", line, do_extract_concepts=False)
                
                # Add concepts
                for c in concepts:
                    # We need to add concept to gravity field
                    # holo.add_text does this if do_extract_concepts=True, but we did extraction outside.
                    # We can manually add to field.
                    c_vec = holo.text_encoder.encode(c)
                    holo.field.add(c, vec=c_vec)
                    
            print(f"Processed batch... {len(holo.store.traces)}/{len(lines)}", end="\r")
            
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nIngestion complete.")
    print(f"Total time: {duration:.4f}s")
    print(f"Avg time per line: {duration/len(lines):.4f}s")
    
    holo.save(store_path)

if __name__ == "__main__":
    kb_file = "data/kbs/physics_biased_kb.txt"
    if len(sys.argv) > 1:
        kb_file = sys.argv[1]
        
    build_kb_concurrent(kb_file, "data/concurrent_store.db")
