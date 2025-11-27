# hologram/retrieval.py
from typing import List, Dict, Any, Set
import numpy as np
from .gravity import GravityField, Probe, cosine

def extract_local_field(field: GravityField, probe: Probe, top_k: int = 10) -> Dict[str, Any]:
    """
    Extract a structured subgraph (Memory Packet) around the probe's trajectory.
    
    Args:
        field: The active GravityField.
        probe: The probe after simulation (contains trajectory).
        top_k: Number of core concepts to retrieve.
        
    Returns:
        A dictionary representing the Memory Packet (nodes, edges, glyphs, etc.)
    """
    if not field.sim.concepts:
        return {"nodes": [], "edges": [], "glyphs": []}

    final_pos = probe.pos
    trajectory = probe.trajectory

    # 1. Identify Core Concepts (nearest to final position)
    # We use the field's search method (FAISS or brute force)
    hits = field.search(final_pos, k=top_k)
    core_names = [name for name, score in hits]
    
    # 2. Identify Peripheral Concepts (along trajectory)
    # Sample points along trajectory to find what we passed by
    peripheral_names: Set[str] = set()
    if len(trajectory) > 1:
        # Check mid-points
        for point in trajectory[::2]: # Skip every other point for speed
            p_hits = field.search(point, k=3)
            for name, _ in p_hits:
                if name not in core_names:
                    peripheral_names.add(name)
    
    all_names = list(set(core_names) | peripheral_names)
    
    # 3. Build Nodes
    nodes = []
    for name in all_names:
        c = field.sim.concepts.get(name)
        if not c: continue
        
        node = {
            "name": name,
            "mass": round(c.mass, 3),
            "age": field.sim.global_step - c.last_reinforced,
            # "vector": c.vec.tolist() # Optional: include if needed for client-side viz
        }
        nodes.append(node)
        
    # 4. Build Edges (Relations between selected nodes)
    edges = []
    for i, n1 in enumerate(all_names):
        for n2 in all_names[i+1:]:
            key = (min(n1, n2), max(n1, n2))
            strength = field.sim.relations.get(key, 0.0)
            
            # Also calculate dynamic tension (distance vs relation)
            # If relation is high but distance is large -> High Tension
            if strength > 0.01:
                c1 = field.sim.concepts[n1]
                c2 = field.sim.concepts[n2]
                dist = 1.0 - cosine(c1.vec, c2.vec)
                tension = dist * strength # Heuristic for tension
                
                edges.append({
                    "a": n1,
                    "b": n2,
                    "relation": round(strength, 3),
                    "tension": round(tension, 3)
                })
                
    # 5. Find Glyph Anchors
    # Glyphs are concepts starting with "glyph:"
    # We check if any glyph is close enough to the final probe position
    glyphs = []
    for name, c in field.sim.concepts.items():
        if name.startswith("glyph:"):
            sim = cosine(final_pos, c.vec)
            if sim > 0.4: # Threshold for anchor relevance
                glyphs.append({
                    "id": name.replace("glyph:", ""),
                    "mass": round(c.mass, 3),
                    "similarity": round(sim, 3)
                })
    
    # Sort glyphs by relevance
    glyphs.sort(key=lambda x: x["similarity"], reverse=True)

    return {
        "nodes": nodes,
        "edges": edges,
        "glyphs": glyphs,
        "trajectory_steps": len(trajectory)
    }
