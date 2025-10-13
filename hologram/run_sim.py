
import json
import numpy as np
from hologram.gravity import GravitySim

# Initialize simulator
sim = GravitySim(dim=128, eta=0.08, alpha_neg=0.25, gamma_decay=0.985)

# Seed groups
concept_groups = {
    "physics": ["gravity", "quantum field", "spacetime", "entropy", "black hole"],
    "symbolic": ["glyph", "anchor", "resonance", "drift", "curvature"],
    "everyday": ["cat", "dog", "coffee", "book", "music"],
    "void": ["the", "of", "and", "a"],
}

for _, items in concept_groups.items():
    for it in items:
        sim.add_concept(it, text=it)

# Reinforce some concepts to increase their 'mass' and cause more drift
for _ in range(3):
    for it in ["gravity", "curvature", "resonance", "entropy", "black hole"]:
        sim.add_concept(it, text=it)

# Decay relations a bit
for _ in range(10):
    sim.step_decay()

# Project to 2D
proj, names = sim.project2d()

# Save a plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6))
plt.scatter(proj[:,0], proj[:,1])
for i, name in enumerate(names):
    plt.annotate(name, (proj[i,0], proj[i,1]))
plt.title("GravitySim: PCA projection of concept space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plot_path = "/mnt/data/gravity_space.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=150)

# Dump JSON
X = sim.get_matrix()
rel, rel_names = sim.relation_matrix()
dump = {
    "names": names,
    "vectors": X.tolist(),
    "relations": {
        "names": rel_names,
        "matrix": rel.tolist(),
    }
}
json_path = "/mnt/data/gravity_space.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(dump, f, ensure_ascii=False, indent=2)

print(plot_path)
print(json_path)
