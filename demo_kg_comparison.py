#!/usr/bin/env python3
"""
Side-by-Side Demo: KG vs Holographic Memory

Shows how the same information behaves differently in each paradigm.
"""

import numpy as np
from hologram.api import Hologram

def traditional_kg_approach():
    """Simulates how a traditional KG would handle the same data."""
    print("="*70)
    print("TRADITIONAL KNOWLEDGE GRAPH APPROACH")
    print("="*70)
    
    # Manually defined triples
    kg = {
        ("Mass", "increases_with", "Velocity"),
        ("Velocity", "limited_by", "SpeedOfLight"),
        ("Energy", "equals", "Mass * c^2"),
        ("Time", "affected_by", "Velocity"),
    }
    
    print("\n[Structure] Explicit edges:")
    for subj, pred, obj in sorted(kg):
        print(f"  {subj} --[{pred}]--> {obj}")
    
    print("\n[Query] 'What happens to mass at high velocity?'")
    print("  Pattern match: (Mass, ?, Velocity)")
    result = [edge for edge in kg if edge[0] == "Mass" and "Velocity" in edge]
    if result:
        print(f"  Found: {result[0]}")
        print("  Interpretation: Mass increases with Velocity")
    
    print("\n[Add conflicting info] 'Mass does NOT change with velocity (Galilean)'")
    print("  Options:")
    print("    1. Add: (Mass, NOT_increases_with, Velocity)")
    print("    2. Or: Create separate context/version")
    print("  Problem: How to query when both edges exist?")
    
    print("\n[Forgetting]")
    print("  Requires: Manual deletion or archival")
    print("  No natural decay mechanism")
    
    print("\n" + "="*70)

def holographic_memory_approach():
    """Shows how holographic memory handles the same data."""
    print("\n" + "="*70)
    print("HOLOGRAPHIC MEMORY APPROACH")
    print("="*70)
    
    hg = Hologram.init(use_clip=False)
    hg.glyphs.create("physics", title="Physics Knowledge")
    
    print("\n[Structure] Emergent field:")
    
    # Add same facts as natural language
    statements = [
        "Mass increases with velocity",
        "Velocity is limited by the speed of light",
        "Energy equals mass times c squared",
        "Time is affected by velocity",
    ]
    
    ids = []
    for stmt in statements:
        id_ = hg.add_text("physics", stmt)
        ids.append(id_)
    
    print("  (Concepts automatically embedded and positioned in vector space)")
    
    # Show emergent clustering
    print("\n[Gravitational connections]")
    for i, stmt in enumerate(statements[:2]):
        id_ = ids[i]
        related = []
        for (a, b), strength in hg.store.sim.relations.items():
            if id_ in (a, b) and strength > 0.1:
                other_id = b if a == id_ else a
                if other_id in hg.store.traces:
                    related.append((hg.store.traces[other_id].content, strength))
        
        if related:
            print(f"\n  '{stmt}'")
            print(f"    gravitationally connected to:")
            for content, strength in sorted(related, key=lambda x: -x[1])[:2]:
                print(f"      [{strength:.3f}] {content}")
    
    print("\n[Query] 'What happens to mass at high velocity?'")
    hits = hg.search_text("mass at high velocity", top_k=2)
    print("  Vector similarity match:")
    for trace, score in hits:
        print(f"    [{score:.3f}] {trace.content}")
    
    print("\n[Add conflicting info] 'Mass does NOT change with velocity'")
    neg_id = hg.add_text("physics", "Mass does NOT change with velocity")
    print(f"  Negation detected: {hg.store.sim.concepts[neg_id].negation}")
    print("  Effect: Creates REPULSION from 'Mass increases with velocity'")
    print("  Result: Both statements coexist in field with tension")
    
    # Show they've moved apart
    original_vec = hg.store.sim.concepts[ids[0]].vec
    negation_vec = hg.store.sim.concepts[neg_id].vec
    similarity = np.dot(original_vec, negation_vec)
    print(f"  Similarity after negation: {similarity:.3f} (negative = repelled)")
    
    print("\n[Forgetting] Simulate passage of time...")
    print("  Reinforcing 'Energy equals mass...' but not others")
    
    # Reinforce one concept
    hg.add_text("physics", "Energy equals mass times c squared")
    
    # Add many new concepts to advance time
    for i in range(10):
        hg.add_text("physics", f"Unrelated concept {i}")
    
    # Apply decay
    for _ in range(5):
        hg.decay(steps=1)
    
    # Check masses
    mass_stmt_mass = hg.store.sim.concepts[ids[0]].mass
    energy_stmt_mass = hg.store.sim.concepts[ids[2]].mass
    
    print(f"\n  After decay:")
    print(f"    'Mass increases...' (unreinforced): mass={mass_stmt_mass:.3f}")
    print(f"    'Energy equals...' (reinforced): mass={energy_stmt_mass:.3f}")
    print(f"  → Unreinforced concepts naturally fade!")
    
    print("\n" + "="*70)

def main():
    traditional_kg_approach()
    holographic_memory_approach()
    
    print("\n" + "="*70)
    print("KEY DIFFERENCES DEMONSTRATED")
    print("="*70)
    print("""
1. STRUCTURE:
   KG:   Explicit edges, manually defined
   Holo: Emergent connections, self-organizing

2. CONTRADICTION HANDLING:
   KG:   Must choose or version
   Holo: Coexistence with tension (repulsion)

3. FORGETTING:
   KG:   Manual deletion required
   Holo: Automatic decay based on usage

4. QUERY:
   KG:   Pattern matching on labeled edges
   Holo: Vector similarity + field traversal

5. EVOLUTION:
   KG:   Static until manually updated
   Holo: Continuous drift and reorg
    """)
    print("="*70)

if __name__ == "__main__":
    main()
