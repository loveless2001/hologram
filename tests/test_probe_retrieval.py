# tests/test_probe_retrieval.py
import numpy as np
from hologram.gravity import Gravity, Concept, TIER_DOMAIN
from hologram.cost_engine import CostEngine
from hologram.probe import ProbeRetriever

def test_probe_avoids_high_cost():
    print("--- Test: Probe Avoids High Cost Nodes ---")
    
    # 1. Setup Gravity Field
    g = Gravity()
    
    # Query Vector
    q_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    
    # Node A: "Trap" 
    # High similarity to query (close to [1,0,0])
    # But High Cost (we will rig neighbors to be chaotic)
    vec_trap = np.array([0.99, 0.1, 0.0], dtype=np.float32) 
    vec_trap /= np.linalg.norm(vec_trap)
    g.add_concept("TRAP", vec=vec_trap, mass=1.0)
    
    # Node B: "Target"
    # Pretty high similarity (slightly less perfect than trap maybe, or equal)
    # Low Cost (stable neighborhood)
    vec_target = np.array([0.98, -0.1, 0.0], dtype=np.float32)
    vec_target /= np.linalg.norm(vec_target)
    g.add_concept("TARGET", vec=vec_target, mass=1.0)
    
    # --- TRAP NEIGHBORHOOD (Cost Maximization) ---
    # 1. Maximize Entropy: High variance in distance
    # 2. Maximize Instability: Opposing directions
    
    # Neighbor 1: Far, Pulls Left
    v1 = vec_trap + np.array([0, 0.8, 0])
    v1 /= np.linalg.norm(v1)
    g.add_concept("trap_n1", vec=v1, mass=0.1)
    g.relations[("TRAP", "trap_n1")] = 0.9

    # Neighbor 2: Near, Pulls Right
    v2 = vec_trap + np.array([0, -0.05, 0])
    v2 /= np.linalg.norm(v2)
    g.add_concept("trap_n2", vec=v2, mass=0.1)
    g.relations[("TRAP", "trap_n2")] = 0.9
    
    # Neighbor 3: Far, Pulls Up
    v3 = vec_trap + np.array([0, 0, 0.9])
    v3 /= np.linalg.norm(v3)
    g.add_concept("trap_n3", vec=v3, mass=0.1)
    g.relations[("TRAP", "trap_n3")] = 0.9

    # Neighbor 4: Near, Pulls Down
    v4 = vec_trap + np.array([0, 0, -0.05])
    v4 /= np.linalg.norm(v4)
    g.add_concept("trap_n4", vec=v4, mass=0.1)
    g.relations[("TRAP", "trap_n4")] = 0.9
        
    # --- TARGET NEIGHBORHOOD (Cost Minimization) ---
    # 1. Minimize Entropy: Uniform distance (StdDev=0)
    # 2. Minimize Instability: Coherent direction (pulling same way)
    
    # All neighbors pull "East" (relative to Target) at distance 0.1
    # Target is at [0.98, -0.1, 0]
    # We add neighbors at Target + [0.1, 0, 0] roughly
    
    direction = np.array([0.1, 0, 0], dtype=np.float32)
    
    for i in range(5):
        # Add tiny jitter to prevent identical vectors (singularities)
        jitter = np.random.randn(3) * 0.001
        v = vec_target + direction + jitter
        v /= np.linalg.norm(v)
        
        g.add_concept(f"target_friend_{i}", vec=v, mass=0.1)
        g.relations[("TARGET", f"target_friend_{i}")] = 0.8

    # 2. Run Probe
    cost_engine = CostEngine()
    
    # Check cost signals first to verify setup
    report_trap = cost_engine.evaluate_node("TRAP", g)
    report_target = cost_engine.evaluate_node("TARGET", g)
    
    print(f"TRAP Cost: {report_trap.total:.3f} (Entropy: {report_trap.entropy:.3f})")
    print(f"TARGET Cost: {report_target.total:.3f} (Entropy: {report_target.entropy:.3f})")
    
    assert report_trap.total > report_target.total, "Setup failed: Trap should have higher cost than Target"
    
    retriever = ProbeRetriever(g, cost_engine)
    tree = retriever.retrieve_tree(q_vec, top_k_seeds=10, max_depth=2, final_k=20)
    
    # 3. Analyze Results
    print("\nRetrieval Tree Nodes:")
    found_trap = False
    found_target = False
    
    for nid, node in tree.nodes.items():
        print(f" - {nid}: Energy={node.energy_from_query:.3f}, Path={node.path_energy:.3f}")
        if nid == "TRAP": found_trap = True
        if nid == "TARGET": found_target = True
        
    # Ideally, TRAP energy should be higher than TARGET energy
    if found_trap and found_target:
        node_trap = tree.nodes["TRAP"]
        node_target = tree.nodes["TARGET"]
        print(f"\nEnergy Comparison: TRAP ({node_trap.energy_from_query:.3f}) vs TARGET ({node_target.energy_from_query:.3f})")
        assert node_trap.energy_from_query > node_target.energy_from_query, "Probe failed: Trap should have higher energy"
        print("SUCCESS: Target has lower energy than Trap.")
    elif found_target and not found_trap:
        print("SUCCESS: Trap was completely avoided!")
    elif found_trap and not found_target:
        print("FAILURE: Trap found but Target missed.")
    else:
        print("FAILURE: Neither found (maybe top_k too small?)")

if __name__ == "__main__":
    test_probe_avoids_high_cost()
