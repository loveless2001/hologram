import numpy as np
from hologram.mg_scorer import mg_score

def demo():
    rng = np.random.default_rng(42)
    
    print("--- 1. High Coherence Cluster ---")
    # Tight cluster around [1, 1, 1, 1, 1]
    base = np.ones(5)
    cluster = base + 0.01 * rng.normal(size=(5, 5))
    s1 = mg_score(cluster)
    print(f"Coherence: {s1.coherence:.4f} (Expected > 0.9)")
    print(f"Entropy:   {s1.entropy:.4f}   (Expected Low)")
    print(f"Risk:      {s1.collapse_risk:.4f}")
    
    print("\n--- 2. Low Coherence (Random) ---")
    # Random vectors
    random_vecs = rng.normal(size=(5, 5))
    s2 = mg_score(random_vecs)
    print(f"Coherence: {s2.coherence:.4f} (Expected < 0.5)")
    print(f"Entropy:   {s2.entropy:.4f}   (Expected High)")
    print(f"Risk:      {s2.collapse_risk:.4f}")

    print("\n--- 3. Linear Trajectory ---")
    # v0, v1, v2 in a line
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([2.0, 0.0, 0.0])
    traj = [v0, v1, v2]
    s3 = mg_score(traj)
    print(f"Curvature: {s3.curvature:.4f} (Expected 0.5 for equal steps)")
    
    print("\n--- 4. Sharp Turn ---")
    # v0, v1, v2 with 90 deg turn
    v0 = np.array([0.0, 0.0, 0.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 1.0, 0.0])
    traj_turn = [v0, v1, v2]
    s4 = mg_score(traj_turn)
    print(f"Curvature: {s4.curvature:.4f} (Expected ~0.707)")

if __name__ == "__main__":
    demo()
