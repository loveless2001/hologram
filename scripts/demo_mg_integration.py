#!/usr/bin/env python3
"""
MG Scorer Integration Demo

This script demonstrates the full capabilities of the MG Scorer
integrated with the Hologram memory system.
"""

from hologram import Hologram
import numpy as np

def print_score(name, score):
    """Pretty print an MGScore."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Coherence:     {score.coherence:.4f}  {'✓' if score.coherence > 0.7 else '⚠'}")
    print(f"  Curvature:     {score.curvature:.4f}")
    print(f"  Entropy:       {score.entropy:.4f}")
    print(f"  Collapse Risk: {score.collapse_risk:.4f}  {'✓' if score.collapse_risk < 0.2 else '⚠'}")
    print(f"  Gradient Norm: {np.linalg.norm(score.gradient):.4f}")

def demo_llm_validation():
    """Demo: Validate LLM output quality."""
    print("\n" + "🤖 DEMO 1: LLM Output Validation".center(60, "="))
    
    hg = Hologram.init(encoder_mode="minilm", use_gravity=False)
    
    # Good output: coherent paragraph
    good_output = [
        "Artificial intelligence is transforming modern technology.",
        "Machine learning enables computers to learn from data.",
        "Deep learning uses neural networks for pattern recognition.",
        "These advances are revolutionizing many industries."
    ]
    
    # Bad output: hallucination/contradiction
    bad_output = [
        "The Eiffel Tower was built in 1889 in Paris.",
        "However, it was actually constructed in London in 1850.",
        "The tower is made of bamboo and reaches 50 feet.",
        "It's the smallest structure in the world."
    ]
    
    score_good = hg.score_text(good_output)
    score_bad = hg.score_text(bad_output)
    
    print_score("✓ Good LLM Output", score_good)
    print_score("✗ Hallucinating LLM Output", score_bad)
    
    # Quality gate
    if score_bad.collapse_risk > 0.2:
        print("\n⚠  ALERT: Output failed quality gate (high collapse risk)")

def demo_topic_drift():
    """Demo: Detect topic drift in conversation."""
    print("\n" + "💬 DEMO 2: Topic Drift Detection".center(60, "="))
    
    hg = Hologram.init(encoder_mode="minilm", use_gravity=False)
    
    # Focused conversation
    focused = [
        "Let's finalize the project proposal today.",
        "The deadline is Friday at 5 PM.",
        "We need to include the budget section.",
        "The proposal should be sent to the client."
    ]
    
    # Drifting conversation
    drifting = [
        "Let's finalize the project proposal today.",
        "Speaking of today, it's my birthday!",
        "Birthdays make me think about getting older.",
        "Aging reminds me of that movie we watched."
    ]
    
    score_focused = hg.score_text(focused)
    score_drifting = hg.score_text(drifting)
    
    print_score("✓ Focused Conversation", score_focused)
    print_score("⇝ Drifting Conversation", score_drifting)
    
    if score_drifting.curvature < 0.6:
        print(f"\n⚠  DRIFT DETECTED: Curvature dropped to {score_drifting.curvature:.2f}")

def demo_memory_health():
    """Demo: Monitor memory field health."""
    print("\n" + "🧠 DEMO 3: Memory Field Health Check".center(60, "="))
    
    hg = Hologram.init(encoder_mode="minilm", use_gravity=True)
    
    # Add some coherent knowledge
    hg.glyphs.create("physics", title="Physics")
    hg.add_text("physics", "Newton's laws describe motion and forces.")
    hg.add_text("physics", "F = ma is the second law of motion.")
    hg.add_text("physics", "Gravity is an attractive force between masses.")
    
    # Add some random noise
    hg.glyphs.create("noise", title="Noise")
    hg.add_text("noise", "Random unrelated concept alpha.")
    hg.add_text("noise", "Completely different topic beta.")
    
    # Score the physics cluster
    physics_traces = [t.trace_id for t in hg.store.traces.values() 
                      if 'physics' in t.content.lower()]
    
    if physics_traces:
        score_physics = hg.score_trace(physics_traces[:5])  # First 5
        print_score("Physics Concept Cluster", score_physics)
        
        if score_physics.coherence > 0.6:
            print("\n✓ Memory cluster is healthy and coherent")
        else:
            print("\n⚠  Memory cluster may need cleanup")

def demo_retrieval_quality():
    """Demo: Quality gate for retrieval results."""
    print("\n" + "🔍 DEMO 4: Retrieval Quality Gate".center(60, "="))
    
    hg = Hologram.init(encoder_mode="minilm", use_gravity=False)
    
    # Build small KB
    hg.glyphs.create("science", title="Science")
    hg.add_text("science", "Photosynthesis converts sunlight to energy.")
    hg.add_text("science", "Plants use chlorophyll to absorb light.")
    hg.add_text("science", "This process sustains life on Earth.")
    hg.add_text("science", "Bananas are yellow.")  # Outlier
    
    # Search
    results = hg.search_text("How do plants produce energy?", top_k=4)
    # Results are (trace, score) tuples
    result_ids = [r[0].trace_id for r in results]
    
    score = hg.score_trace(result_ids)
    print_score("Retrieval Results", score)
    
    if score.coherence > 0.5:
        print("\n✓ Retrieval results are coherent")
    else:
        print("\n⚠  Retrieval results may be noisy - consider re-ranking")

if __name__ == "__main__":
    print("\n" + "🧲 MG Scorer Integration Demo".center(60, "=") + "\n")
    
    demo_llm_validation()
    demo_topic_drift()
    demo_memory_health()
    demo_retrieval_quality()
    
    print("\n" + "="*60)
    print("  Demo Complete!")
    print("="*60 + "\n")
