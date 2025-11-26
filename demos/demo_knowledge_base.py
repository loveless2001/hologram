#!/usr/bin/env python3
"""
Knowledge Base Demo: Special Relativity

Demonstrates using holographic memory as a conceptual knowledge base.
Tests both retrieval and reasoning capabilities.
"""

import numpy as np
from hologram.api import Hologram

def main():
    print("="*70)
    print("HOLOGRAPHIC KNOWLEDGE BASE: Special Relativity")
    print("="*70)
    
    hg = Hologram.init(use_clip=False)
    hg.glyphs.create("relativity", title="Special Relativity")
    
    # Build knowledge base with key concepts and relationships
    print("\n[1] Building knowledge base...")
    
    # Core concepts
    concepts = [
        "Speed of light is constant in all reference frames",
        "Light travels at 299,792,458 meters per second",
        "Speed of light is denoted by the symbol c",
        "Nothing can travel faster than the speed of light",
        
        # Time dilation
        "As velocity approaches c, time dilation becomes extreme",
        "Time dilation means time passes slower for moving objects",
        "At high velocities, clocks tick slower",
        
        # Mass-energy
        "Mass increases as velocity approaches c",
        "Energy and mass are equivalent: E equals mc squared",
        "Accelerating to c requires infinite energy",
        
        # Length contraction
        "Objects appear shorter in the direction of motion at high speeds",
        "Length contraction occurs at relativistic speeds",
        
        # Practical limits
        "Massive objects cannot reach the speed of light",
        "Only massless particles like photons can travel at c",
    ]
    
    for i, statement in enumerate(concepts):
        hg.add_text("relativity", statement)
        if (i + 1) % 5 == 0:
            print(f"  Added {i + 1} statements...")
    
    print(f"  Total: {len(concepts)} statements stored\n")
    
    # Test queries
    queries = [
        "what happens when an object accelerates towards speed of light",
        "why can't we travel faster than light",
        "how does time behave at high speeds",
        "what is the relationship between mass and energy"
    ]
    
    print("="*70)
    print("[2] Testing Query Resolution")
    print("="*70)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 70)
        
        # Retrieve top relevant statements
        hits = hg.search_text(query, top_k=3)
        
        if hits:
            print("Retrieved knowledge:")
            for i, (trace, score) in enumerate(hits, 1):
                print(f"  {i}. [{score:.3f}] {trace.content}")
        else:
            print("  No relevant knowledge found.")
    
    print("\n" + "="*70)
    print("[3] Analysis: What Works & What's Missing")
    print("="*70)
    
    print("\n✓ CURRENT CAPABILITIES:")
    print("  • Semantic retrieval of related statements")
    print("  • Finds concepts with similar embeddings to query")
    print("  • Returns ranked list of relevant knowledge")
    
    print("\n✗ LIMITATIONS (what's not implemented yet):")
    print("  • No inference/reasoning over multiple statements")
    print("  • Cannot chain concepts together (A→B, B→C ∴ A→C)")
    print("  • No natural language answer generation")
    print("  • Doesn't understand causal relationships explicitly")
    
    print("\n💡 WHAT WE COULD ADD:")
    print("  1. Relation extraction: identify [subject, predicate, object] triples")
    print("  2. Reasoning engine: traverse concept graph to answer questions")
    print("  3. Answer synthesis: combine retrieved statements into coherent response")
    print("  4. Use gravity field relations as implicit reasoning paths")
    
    print("\n" + "="*70)
    print("[4] Prototype: Simple Inference via Concept Bridging")
    print("="*70)
    
    # Demonstrate how gravity field could enable reasoning
    print("\nLet's trace concept connections in the gravity simulation:")
    
    # Find concepts related to "accelerate toward c"
    query = "accelerate toward speed of light"
    hits = hg.search_text(query, top_k=5)
    
    print(f"\nQuery: '{query}'")
    print("\nDirect matches:")
    for i, (trace, score) in enumerate(hits[:3], 1):
        print(f"  {i}. {trace.content[:60]}...")
    
    # Check what concepts are gravitationally connected
    if hits:
        top_trace_id = hits[0][0].trace_id
        
        # Look at gravity field relations
        related_concepts = []
        for (a, b), strength in hg.store.sim.relations.items():
            if top_trace_id in (a, b) and strength > 0.1:
                other = b if a == top_trace_id else a
                if other in hg.store.traces:
                    related_concepts.append((hg.store.traces[other].content, strength))
        
        if related_concepts:
            print("\nGravitationally connected concepts (could enable reasoning):")
            for content, strength in sorted(related_concepts, key=lambda x: -x[1])[:3]:
                print(f"  [{strength:.3f}] {content[:60]}...")
    
    print("\n" + "="*70)
    print("CONCLUSION:")
    print("The system currently works as a semantic knowledge retrieval engine.")
    print("With additional inference logic on top of the gravity field, it could")
    print("perform reasoning and answer synthesis. The gravitational connections")
    print("between concepts provide a natural 'reasoning graph'.")
    print("="*70)
    
    # Save for inspection
    hg.save("relativity_kb.json")
    print("\n💾 Knowledge base saved to: relativity_kb.json")

if __name__ == "__main__":
    main()
