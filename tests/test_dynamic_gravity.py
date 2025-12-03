#!/usr/bin/env python3
"""
Test the Dynamic Gravity System: Auto-Fusion, Mass Calibration, and Auto-Mitosis.
"""
import sys
import os
import shutil
import numpy as np
import logging

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hologram import Hologram

# Configure logging to capture the streaming events
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_auto_fusion():
    print("\n" + "="*80)
    print("TEST 1: Auto-Fusion (Standard)")
    print("="*80)
    
    # Initialize
    h = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    # 1. Add canonical concept
    print("Adding 'Quantum Field Theory'...")
    h.add_text("c1", "Quantum Field Theory")
    
    # 2. Add a very close variant (should fuse immediately via fuzzy resolution or dynamics)
    print("Adding 'Quantum field theory' (case variant)...")
    h.add_text("c2", "Quantum field theory")
    
    # Check state
    concepts = h.field.sim.concepts
    active_concepts = [c for c in concepts.values() if c.canonical_id is None]
    
    print(f"Active concepts: {len(active_concepts)}")
    for c in active_concepts:
        print(f"  - {c.name}: Mass={c.mass:.2f}")
        
    if len(active_concepts) == 1 and active_concepts[0].mass >= 2.0:
        print("✅ Standard Fusion Passed")
    else:
        print("❌ Standard Fusion Failed")
        return False
        
    return True

def test_black_hole_effect():
    print("\n" + "="*80)
    print("TEST 2: Mass Calibration (Black Hole Effect)")
    print("="*80)
    
    h = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    # 1. Create a MASSIVE concept
    print("Creating massive concept 'Gravity' (Mass=50.0)...")
    h.add_text("g1", "Gravity")
    # Artificially boost mass to simulate a "Black Hole"
    g_id = list(h.field.sim.concepts.keys())[0]
    h.field.sim.concepts[g_id].mass = 50.0
    
    # 2. Add a somewhat distant variant
    # "Gravitational pull" might have sim ~0.7-0.8 with "Gravity"
    # Standard threshold is 0.85. 
    # With mass 50, threshold should drop: 0.85 - (log(50)*0.02) ≈ 0.85 - (3.9*0.02) ≈ 0.85 - 0.08 = 0.77
    print("Adding 'Gravitational pull'...")
    h.add_text("g2", "Gravitational pull")
    
    # Trigger dynamics explicitly to be sure
    h.field.sim.step_dynamics()
    
    # Check if fused
    concepts = h.field.sim.concepts
    active_concepts = [c for c in concepts.values() if c.canonical_id is None]
    
    print(f"Active concepts: {len(active_concepts)}")
    for c in active_concepts:
        print(f"  - {c.name}: Mass={c.mass:.2f}")
        
    if len(active_concepts) == 1:
        print("✅ Black Hole Effect Passed (Distant concept absorbed)")
    else:
        print("❌ Black Hole Effect Failed (Distant concept NOT absorbed)")
        # Check similarity to debug
        vec1 = h.field.sim.concepts[g_id].vec
        # Find the other one
        other_id = [k for k in concepts if k != g_id][0]
        vec2 = h.field.sim.concepts[other_id].vec
        sim = np.dot(vec1, vec2)
        print(f"   Similarity was: {sim:.4f}")
        return False

    return True

def test_auto_mitosis():
    print("\n" + "="*80)
    print("TEST 3: Auto-Mitosis")
    print("="*80)
    
    h = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    # 1. Create polysemous concept "Bank"
    print("Adding 'Bank'...")
    h.add_text("bank", "Bank")
    bank_id = list(h.field.sim.concepts.keys())[0]
    
    # 2. Add conflicting neighbors
    print("Adding River context...")
    h.add_text("river1", "The river bank is muddy")
    h.add_text("river2", "Water flows along the bank")
    
    print("Adding Finance context...")
    h.add_text("finance1", "I went to the bank to deposit money")
    h.add_text("finance2", "The bank loan interest rate is high")
    
    # Manually create strong relations to force tension
    # (In real usage, relations build up over time/co-occurrence)
    # We need to find the IDs of these new texts
    river_ids = []
    finance_ids = []
    
    for tid, tr in h.store.traces.items():
        if "river" in tr.content.lower() or "water" in tr.content.lower():
            river_ids.append(tid)
        elif "money" in tr.content.lower() or "loan" in tr.content.lower():
            finance_ids.append(tid)
            
    # Link them to Bank
    for rid in river_ids:
        key = (min(bank_id, rid), max(bank_id, rid))
        h.field.sim.relations[key] = 0.9 # Strong link
        
    for fid in finance_ids:
        key = (min(bank_id, fid), max(bank_id, fid))
        h.field.sim.relations[key] = 0.9 # Strong link
        
    # 3. Trigger Dynamics
    print("Triggering dynamics...")
    # Mitosis requires clustering. We need to make sure the vectors pull "Bank" in different directions?
    # Actually, check_mitosis uses the vectors of the NEIGHBORS to cluster them.
    # So if river_ids vectors and finance_ids vectors are distinct, it should split.
    
    h.field.sim.step_dynamics()
    
    # Check results
    concepts = h.field.sim.concepts
    
    # "Bank" (original ID) should be gone or aliased?
    # The code deletes the original and creates _1 and _2
    
    if bank_id not in concepts:
        print("✅ Original 'Bank' concept removed")
    else:
        print("❌ Original 'Bank' concept still exists")
        return False
        
    # Check for siblings
    siblings = [k for k in concepts if "_1" in k or "_2" in k]
    print(f"Siblings found: {siblings}")
    
    if len(siblings) >= 2:
        print("✅ Auto-Mitosis Passed (Split into siblings)")
    else:
        print("❌ Auto-Mitosis Failed")
        return False
        
    return True

if __name__ == "__main__":
    p1 = test_auto_fusion()
    p2 = test_black_hole_effect()
    p3 = test_auto_mitosis()
    
    if p1 and p2 and p3:
        print("\n" + "="*80)
        print("🎉 ALL DYNAMIC GRAVITY TESTS PASSED")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("💥 SOME TESTS FAILED")
        print("="*80)
        sys.exit(1)
