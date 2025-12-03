import sys
import os
import numpy as np
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.gravity import Gravity, TIER_DOMAIN, TIER_SYSTEM, Concept
from hologram.system_kb import get_system_concepts
from hologram.text_utils import extract_concepts

def print_header(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def report_concepts(g: Gravity):
    print("\n--- Concept Report ---")
    by_tier = {TIER_DOMAIN: [], TIER_SYSTEM: [], 3: []}
    
    for name, c in g.concepts.items():
        if c.canonical_id: continue # Skip aliases
        by_tier.setdefault(c.tier, []).append(c)
        
    for tier, label in [(TIER_DOMAIN, "🟦 TIER 1 (Domain)"), (TIER_SYSTEM, "🟧 TIER 2 (System)"), (3, "🟥 TIER 3 (Meta)")]:
        concepts = by_tier.get(tier, [])
        print(f"\n{label} Concepts ({len(concepts)}):")
        for c in concepts:
            print(f"  - {c.name:<30} (mass={c.mass:.1f}, proj={c.project}, orig={c.origin})")

def run_verification():
    print_header("VERIFYING TIER SYSTEM")
    
    g = Gravity(dim=64)
    
    # 1. Ingest System Concepts (Tier 2)
    print("Step 1: Ingesting System Concepts...")
    sys_concepts = ["system:gravity", "system:fusion", "system:mitosis"]
    for name in sys_concepts:
        g.add_concept(name, tier=TIER_SYSTEM, project="hologram", origin="system_design", mass=5.0)
        
    # 2. Add Domain Concepts (Tier 1)
    print("Step 2: Adding Domain Concepts...")
    # Group A: Project 1
    g.add_concept("p1:concept_a", tier=TIER_DOMAIN, project="p1", origin="kb", mass=1.0)
    g.add_concept("p1:concept_b", tier=TIER_DOMAIN, project="p1", origin="kb", mass=1.0) # Close to A
    
    # Group B: Project 2
    g.add_concept("p2:concept_x", tier=TIER_DOMAIN, project="p2", origin="kb", mass=1.0)
    
    # Make p1:concept_b very close to p1:concept_a (force fusion)
    g.concepts["p1:concept_b"].vec = g.concepts["p1:concept_a"].vec.copy()
    
    # Make p2:concept_x close to p1:concept_a (should NOT fuse)
    g.concepts["p2:concept_x"].vec = g.concepts["p1:concept_a"].vec.copy()
    
    # Make system:gravity close to p1:concept_a (should NOT fuse)
    g.concepts["system:gravity"].vec = g.concepts["p1:concept_a"].vec.copy()
    
    report_concepts(g)
    
    # 3. Trigger Dynamics
    print_header("Step 3: Triggering Dynamics (Fusion/Mitosis)")
    g.step_dynamics()
    
    print("\n--- Post-Dynamics Report ---")
    report_concepts(g)
    
    # Verification
    print("\n--- Verification Results ---")
    
    # Check 1: p1:concept_b should be fused into p1:concept_a (or vice versa)
    c_a = g.concepts["p1:concept_a"]
    c_b = g.concepts["p1:concept_b"]
    
    print(f"DEBUG: a.canon={c_a.canonical_id}, b.canon={c_b.canonical_id}")
    
    if c_b.canonical_id == "p1:concept_a" or c_a.canonical_id == "p1:concept_b":
        print("✅ Tier 1 Same-Project Fusion: SUCCESS")
    else:
        print("❌ Tier 1 Same-Project Fusion: FAILED")
        
    # Check 2: p2:concept_x should NOT be fused
    c_x = g.concepts["p2:concept_x"]
    if c_x.canonical_id is None:
        print("✅ Cross-Project Protection: SUCCESS")
    else:
        print(f"❌ Cross-Project Protection: FAILED (fused into {c_x.canonical_id})")
        
    # Check 3: system:gravity should NOT be fused
    c_sys = g.concepts["system:gravity"]
    if c_sys.canonical_id is None:
        print("✅ System Tier Protection: SUCCESS")
    else:
        print(f"❌ System Tier Protection: FAILED (fused into {c_sys.canonical_id})")

    # 4. Mitosis Test
    print_header("Step 4: Mitosis Verification")
    # Create a massive concept with tension
    g.add_concept("p1:massive", tier=TIER_DOMAIN, project="p1", mass=10.0)
    
    # Add neighbors to create tension (Needs at least 3 neighbors)
    n1_vec = np.random.rand(64)
    n2_vec = -n1_vec # Opposite
    n3_vec = n1_vec + np.random.normal(0, 0.1, 64) # Close to n1
    
    g.add_concept("n1", vec=n1_vec, tier=TIER_DOMAIN, project="p1")
    g.add_concept("n2", vec=n2_vec, tier=TIER_DOMAIN, project="p1")
    g.add_concept("n3", vec=n3_vec, tier=TIER_DOMAIN, project="p1")
    
    # Link them
    g.relations[("n1", "p1:massive")] = 0.9
    g.relations[("n2", "p1:massive")] = 0.9
    g.relations[("n3", "p1:massive")] = 0.9
    
    # Try mitosis on domain concept
    if g.check_mitosis("p1:massive", cooldown_steps=0):
        print("✅ Tier 1 Mitosis: SUCCESS")
    else:
        print("❌ Tier 1 Mitosis: FAILED (or conditions not met)")
        
    # Try mitosis on system concept
    g.relations[("n1", "system:fusion")] = 0.9
    g.relations[("n2", "system:fusion")] = 0.9
    g.relations[("n3", "system:fusion")] = 0.9
    if g.check_mitosis("system:fusion", cooldown_steps=0):
        print("❌ System Tier Mitosis: FAILED (Should be blocked)")
    else:
        print("✅ System Tier Mitosis: BLOCKED (Correct)")

if __name__ == "__main__":
    run_verification()
