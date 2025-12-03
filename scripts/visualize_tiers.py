import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hologram.gravity import Gravity, TIER_DOMAIN, TIER_SYSTEM

def visualize_field():
    print("Generating Tier Visualization...")
    g = Gravity(dim=64)
    
    # Create random concepts
    # Tier 1: Blue
    for i in range(20):
        vec = np.random.randn(64)
        vec /= np.linalg.norm(vec)
        g.add_concept(f"d_{i}", vec=vec, tier=TIER_DOMAIN)
        
    # Tier 2: Orange (Fixed Anchors)
    for i in range(5):
        vec = np.random.randn(64)
        vec /= np.linalg.norm(vec)
        g.add_concept(f"sys_{i}", vec=vec, tier=TIER_SYSTEM, mass=5.0)
        
    # Project to 2D
    proj, names = g.project2d()
    
    # Plot
    plt.figure(figsize=(10, 8))
    
    for i, name in enumerate(names):
        c = g.concepts[name]
        x, y = proj[i]
        
        color = 'blue' if c.tier == TIER_DOMAIN else 'orange'
        marker = 'o' if c.tier == TIER_DOMAIN else 's'
        size = c.mass * 50
        
        plt.scatter(x, y, c=color, marker=marker, s=size, alpha=0.6)
        plt.text(x, y, name, fontsize=8)
        
    plt.title("Hologram Field: Tier 1 (Blue) vs Tier 2 (Orange)")
    plt.grid(True, alpha=0.3)
    
    output_path = "tier_visualization.png"
    plt.savefig(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    visualize_field()
