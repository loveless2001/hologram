# scripts/seed_relativity.py
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from hologram.api import Hologram

def seed():
    print("Initializing Hologram with Gravity...")
    # Use hashing for speed, but enable gravity
    memory = Hologram.init(use_clip=False, use_gravity=True)
    
    # Create a glyph for the domain
    memory.glyphs.create("physics", title="Physics Concepts")
    
    # Use full sentences to test concept decomposition
    sentences = [
        "Special Relativity describes how time dilation occurs near the speed of light.",
        "Length contraction is observed when an object moves at a significant fraction of the speed of light.",
        "The speed of light is constant in all inertial frames of reference.",
        "The twin paradox illustrates the effects of time dilation in spacetime.",
        "Mass-energy equivalence states that energy equals mass times the speed of light squared.",
        "Lorentz transformations connect space and time coordinates between inertial frames."
    ]
    
    unrelated = [
        "Apple pie is a popular dessert in America.",
        "Baseball is played with a bat, a ball, and gloves.",
        "Jazz music originated in New Orleans.",
        "The subway system provides efficient transportation in the city."
    ]
    
    print("Adding Relativity sentences...")
    for s in sentences:
        print(f"  - {s}")
        # In the script we just add raw text, but the API server will decompose it
        memory.add_text("physics", s)
        
    print("Adding unrelated sentences...")
    for s in unrelated:
        print(f"  - {s}")
        memory.add_text("physics", s)
        
    # Force decay steps to let gravity work
    print("Simulating gravity (decay steps)...")
    memory.decay(steps=5)
    
    print("Saving state...")
    # For this demo, we just keep it in memory if running as script, 
    # but to persist for the API server, we need to save it.
    # However, the API server loads from 'data/kbs'.
    # Let's write a text file to data/kbs/relativity.txt so the API server can load it.
    
    kb_path = Path("data/kbs/relativity.txt")
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    
    content = "\n".join(sentences + unrelated)
    kb_path.write_text(content)
    print(f"Written to {kb_path}")
    print("Done! Start the API server and load 'relativity.txt' KB to see decomposed concepts.")

if __name__ == "__main__":
    seed()
