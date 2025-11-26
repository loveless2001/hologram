#!/usr/bin/env python3
"""
Demo: Negation-Aware Memory Gravity

This demonstrates how the holographic memory system handles negation.
Negative statements (e.g., "X is not Y") create repulsive forces instead 
of attractive ones, pushing concepts apart in vector space.
"""

import numpy as np
from hologram.api import Hologram

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def main():
    print("="*60)
    print("NEGATION-AWARE MEMORY GRAVITY DEMO")
    print("="*60)
    
    hg = Hologram.init(use_clip=False)
    hg.glyphs.create("philosophy", title="Philosophical Concepts")
    
    # Example 1: Positive relationship
    print("\n[1] Testing POSITIVE relationship: 'Memory is Gravity'")
    print("-" * 60)
    
    id_mem = hg.add_text("philosophy", "Memory")
    id_grav = hg.add_text("philosophy", "Gravity")
    
    vec_mem = hg.store.sim.concepts[id_mem].vec.copy()
    vec_grav = hg.store.sim.concepts[id_grav].vec.copy()
    dist_before = np.linalg.norm(vec_mem - vec_grav)
    
    print(f"  Initial distance(Memory, Gravity): {dist_before:.4f}")
    
    hg.add_text("philosophy", "Memory is Gravity")
    
    dist_after = np.linalg.norm(hg.store.sim.concepts[id_mem].vec - hg.store.sim.concepts[id_grav].vec)
    print(f"  After 'Memory is Gravity': {dist_after:.4f}")
    print(f"  → Concepts moved {'CLOSER' if dist_after < dist_before else 'APART'} (Δ={dist_after-dist_before:+.4f})")
    
    # Example 2: Negative relationship
    print("\n[2] Testing NEGATIVE relationship: 'Collapse isn't the End'")
    print("-" * 60)
    
    id_col = hg.add_text("philosophy", "Collapse")
    id_end = hg.add_text("philosophy", "End")
    
    vec_col = hg.store.sim.concepts[id_col].vec.copy()
    vec_end = hg.store.sim.concepts[id_end].vec.copy()
    dist_before2 = np.linalg.norm(vec_col - vec_end)
    
    print(f"  Initial distance(Collapse, End): {dist_before2:.4f}")
    
    id_neg = hg.add_text("philosophy", "Collapse isn't the End")
    has_negation = hg.store.sim.concepts[id_neg].negation
    
    dist_after2 = np.linalg.norm(hg.store.sim.concepts[id_col].vec - hg.store.sim.concepts[id_end].vec)
    print(f"  Negation detected: {has_negation}")
    print(f"  After 'Collapse isn't the End': {dist_after2:.4f}")
    print(f"  → Concepts moved {'APART' if dist_after2 > dist_before2 else 'CLOSER'} (Δ={dist_after2-dist_before2:+.4f})")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("✓ Positive statements create ATTRACTION (concepts drift together)")
    print("✓ Negative statements create REPULSION (concepts push apart)")
    print("\nThis allows the system to model both relationships AND negations,")
    print("creating a more nuanced semantic space that respects logical structure.")
    print("="*60)
    
    # Save the memory
    hg.save("philosophy_memory.json")
    print("\n💾 Saved to philosophy_memory.json")

if __name__ == "__main__":
    main()
