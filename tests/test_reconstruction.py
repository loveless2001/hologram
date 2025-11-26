import sys
from pathlib import Path
from hologram.api import Hologram
from hologram.text_utils import extract_concepts

def test_reconstruction():
    print("Initializing Hologram...")
    memory = Hologram.init(use_gravity=True)
    
    kb_path = Path("data/kbs/relativity.txt")
    if not kb_path.exists():
        print(f"Error: {kb_path} not found. Run seed_relativity.py first.")
        return

    print(f"Loading and decomposing KB: {kb_path}")
    with open(kb_path, "r") as f:
        for line in f:
            text = line.strip()
            if text:
                concepts = extract_concepts(text)
                for concept in concepts:
                    memory.add_text("root", concept)
    
    seed_keyword = "speed of light"
    print(f"\n--- Reconstructing knowledge for seed: '{seed_keyword}' ---")
    
    # Search for related concepts
    # We use a higher top_k to see the "neighborhood"
    results = memory.search_text(seed_keyword, top_k=10)
    
    print(f"Found {len(results)} related traces:")
    for trace, score in results:
        print(f"  [{score:.4f}] {trace.content}")

if __name__ == "__main__":
    test_reconstruction()
