# tests/test_glyph_physics.py
import numpy as np
from hologram.api import Hologram

def test_glyph_centroid_and_mass():
    print("--- Testing Glyph Centroid and Mass ---")
    # Init without CLIP for speed (uses hashing encoder)
    holo = Hologram.init(use_clip=False)
    g_id = "🝞"
    holo.glyphs.create(g_id, title="Curvature Anchor")

    v1 = "memory is gravity"
    v2 = "gravity bends concepts toward each other"

    print("Adding trace 1...")
    holo.add_text(g_id, v1)
    print("Adding trace 2...")
    holo.add_text(g_id, v2)

    store = holo.store
    gravity = store.sim

    name = f"glyph:{g_id}"
    assert name in gravity.concepts, f"Glyph concept '{name}' not found in gravity"

    c = gravity.concepts[name]
    print(f"Glyph Mass: {c.mass:.3f}")
    assert c.mass > 1.0, "Glyph mass should be > 1.0"
    assert c.count >= 1, "Glyph count should be >= 1"

    # Centroid approximation check
    vecs = []
    for tid in store.get_glyph(g_id).trace_ids:
        vecs.append(store.get_trace(tid).vec)
    mat = np.stack(vecs, axis=0)
    centroid = mat.mean(axis=0)
    centroid /= (np.linalg.norm(centroid) + 1e-8)

    dot = float(np.dot(centroid, c.vec))
    print(f"Centroid Alignment (Dot Product): {dot:.5f}")
    assert dot > 0.99, "Glyph vector should match centroid of traces"
    print("✓ Centroid and Mass verified")

def test_glyph_mass_growth():
    print("\n--- Testing Glyph Mass Growth ---")
    holo = Hologram.init(use_clip=False)
    g_id = "field"

    holo.glyphs.create(g_id, title="Field")
    
    print("Adding trace 1...")
    holo.add_text(g_id, "field in physics")
    m1 = holo.store.sim.concepts[f"glyph:{g_id}"].mass
    print(f"Mass 1: {m1:.3f}")

    print("Adding trace 2...")
    holo.add_text(g_id, "field of wheat")
    m2 = holo.store.sim.concepts[f"glyph:{g_id}"].mass
    print(f"Mass 2: {m2:.3f}")

    assert m2 > m1, "Mass should increase with more traces"
    print("✓ Mass growth verified")

if __name__ == "__main__":
    test_glyph_centroid_and_mass()
    test_glyph_mass_growth()
