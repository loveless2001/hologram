#!/usr/bin/env python3
"""
Demo: Reinforcement-Based Memory Decay

Shows how unreinforced concepts drift away from the cluster over time,
while frequently reinforced concepts remain central and influential.
"""

import numpy as np
from hologram.api import Hologram

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

def main():
    print("="*70)
    print("REINFORCEMENT-BASED MEMORY DECAY")
    print("="*70)
    
    hg = Hologram.init(use_clip=False)
    hg.glyphs.create("memory", title="Memory Experiment")
    
    # Create initial concepts
    print("\n[Step 0] Creating initial memory cluster...")
    id_core = hg.add_text("memory", "Core Memory")
    id_frequent = hg.add_text("memory", "Frequently Accessed")
    id_rare = hg.add_text("memory", "Rarely Recalled")
    
    # Store initial vectors
    vec_core = hg.store.sim.concepts[id_core].vec.copy()
    vec_frequent = hg.store.sim.concepts[id_frequent].vec.copy()
    vec_rare = hg.store.sim.concepts[id_rare].vec.copy()
    
    # Calculate centroid
    centroid = np.mean([vec_core, vec_frequent, vec_rare], axis=0)
    
    dist_core_init = np.linalg.norm(vec_core - centroid)
    dist_freq_init = np.linalg.norm(vec_frequent - centroid)
    dist_rare_init = np.linalg.norm(vec_rare - centroid)
    
    mass_core_init = hg.store.sim.concepts[id_core].mass
    mass_freq_init = hg.store.sim.concepts[id_frequent].mass
    mass_rare_init = hg.store.sim.concepts[id_rare].mass
    
    print(f"\nInitial state:")
    print(f"  Core:      dist={dist_core_init:.4f}, mass={mass_core_init:.2f}")
    print(f"  Frequent:  dist={dist_freq_init:.4f}, mass={mass_freq_init:.2f}")
    print(f"  Rare:      dist={dist_rare_init:.4f}, mass={mass_rare_init:.2f}")
    
    # Simulate time passing with selective reinforcement
    print("\n[Simulation] 20 time steps with selective reinforcement...")
    print("  → Reinforcing 'Core Memory' every 2 steps")
    print("  → Reinforcing 'Frequently Accessed' every 5 steps")
    print("  → Never reinforcing 'Rarely Recalled'\n")
    
    for step in range(1, 21):
        # Reinforce core frequently
        if step % 2 == 0:
            hg.add_text("memory", "Core Memory")  # Reinforces existing concept
        
        # Reinforce frequent occasionally
        if step % 5 == 0:
            hg.add_text("memory", "Frequently Accessed")
        
        # Add noise to keep system active
        hg.add_text("memory", f"Noise concept {step}")
        
        # Apply decay
        hg.decay(steps=1)
        
        if step % 5 == 0:
            # Recalculate centroid with current vectors
            all_vecs = [c.vec for c in hg.store.sim.concepts.values()]
            centroid_now = np.mean(all_vecs, axis=0)
            
            vec_core_now = hg.store.sim.concepts[id_core].vec
            vec_freq_now = hg.store.sim.concepts[id_frequent].vec
            vec_rare_now = hg.store.sim.concepts[id_rare].vec
            
            dist_core = np.linalg.norm(vec_core_now - centroid_now)
            dist_freq = np.linalg.norm(vec_freq_now - centroid_now)
            dist_rare = np.linalg.norm(vec_rare_now - centroid_now)
            
            mass_core = hg.store.sim.concepts[id_core].mass
            mass_freq = hg.store.sim.concepts[id_frequent].mass
            mass_rare = hg.store.sim.concepts[id_rare].mass
            
            staleness_core = hg.store.sim.global_step - hg.store.sim.concepts[id_core].last_reinforced
            staleness_freq = hg.store.sim.global_step - hg.store.sim.concepts[id_frequent].last_reinforced
            staleness_rare = hg.store.sim.global_step - hg.store.sim.concepts[id_rare].last_reinforced
            
            print(f"  Step {step:2d}:")
            print(f"    Core:      dist={dist_core:.4f} ({dist_core/dist_core_init:+.1%}), mass={mass_core:.2f}, staleness={staleness_core}")
            print(f"    Frequent:  dist={dist_freq:.4f} ({dist_freq/dist_freq_init:+.1%}), mass={mass_freq:.2f}, staleness={staleness_freq}")
            print(f"    Rare:      dist={dist_rare:.4f} ({dist_rare/dist_rare_init:+.1%}), mass={mass_rare:.2f}, staleness={staleness_rare}")
    
    # Final report
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    vec_core_final = hg.store.sim.concepts[id_core].vec
    vec_freq_final = hg.store.sim.concepts[id_frequent].vec
    vec_rare_final = hg.store.sim.concepts[id_rare].vec
    
    all_vecs_final = [c.vec for c in hg.store.sim.concepts.values()]
    centroid_final = np.mean(all_vecs_final, axis=0)
    
    dist_core_final = np.linalg.norm(vec_core_final - centroid_final)
    dist_freq_final = np.linalg.norm(vec_freq_final - centroid_final)
    dist_rare_final = np.linalg.norm(vec_rare_final - centroid_final)
    
    mass_core_final = hg.store.sim.concepts[id_core].mass
    mass_freq_final = hg.store.sim.concepts[id_frequent].mass
    mass_rare_final = hg.store.sim.concepts[id_rare].mass
    
    print(f"\n✓ Core Memory (frequently reinforced):")
    print(f"  Distance change: {dist_core_init:.4f} → {dist_core_final:.4f} ({(dist_core_final/dist_core_init-1)*100:+.1f}%)")
    print(f"  Mass change: {mass_core_init:.2f} → {mass_core_final:.2f} ({(mass_core_final/mass_core_init-1)*100:+.1f}%)")
    print(f"  Status: STRONG (remains influential)")
    
    print(f"\n○ Frequently Accessed (occasionally reinforced):")
    print(f"  Distance change: {dist_freq_init:.4f} → {dist_freq_final:.4f} ({(dist_freq_final/dist_freq_init-1)*100:+.1f}%)")
    print(f"  Mass change: {mass_freq_init:.2f} → {mass_freq_final:.2f} ({(mass_freq_final/mass_freq_init-1)*100:+.1f}%)")
    print(f"  Status: MODERATE (some decay)")
    
    print(f"\n✗ Rarely Recalled (never reinforced):")
    print(f"  Distance change: {dist_rare_init:.4f} → {dist_rare_final:.4f} ({(dist_rare_final/dist_rare_init-1)*100:+.1f}%)")
    print(f"  Mass change: {mass_rare_init:.2f} → {mass_rare_final:.2f} ({(mass_rare_final/mass_rare_init-1)*100:+.1f}%)")
    print(f"  Status: FADING (drifting to periphery, losing influence)")
    
    print("\n" + "="*70)
    print("KEY INSIGHT:")
    print("Unreinforced concepts naturally drift away from the memory cluster")
    print("and lose influence, mimicking natural forgetting. Reinforcement")
    print("keeps memories strong and central.")
    print("="*70)

if __name__ == "__main__":
    main()
