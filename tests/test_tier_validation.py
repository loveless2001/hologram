import pytest
import numpy as np
from hologram.gravity import Gravity, Concept, TIER_DOMAIN, TIER_SYSTEM, can_interact, is_protected_namespace

@pytest.fixture
def gravity():
    return Gravity(dim=4)

def test_tier_constants():
    assert TIER_DOMAIN == 1
    assert TIER_SYSTEM == 2

def test_is_protected_namespace():
    assert is_protected_namespace("system:foo")
    assert is_protected_namespace("meta:bar")
    assert is_protected_namespace("hologram:baz")
    assert is_protected_namespace("architecture:qux")
    assert not is_protected_namespace("user:foo")
    assert not is_protected_namespace("concept:bar")

def test_can_interact_valid():
    c1 = Concept("a", np.zeros(4), tier=TIER_DOMAIN, project="p1", origin="kb")
    c2 = Concept("b", np.zeros(4), tier=TIER_DOMAIN, project="p1", origin="kb")
    assert can_interact(c1, c2)

def test_can_interact_tier_mismatch():
    c1 = Concept("a", np.zeros(4), tier=TIER_DOMAIN)
    c2 = Concept("b", np.zeros(4), tier=TIER_SYSTEM)
    assert not can_interact(c1, c2)

def test_can_interact_project_mismatch():
    c1 = Concept("a", np.zeros(4), tier=TIER_DOMAIN, project="p1")
    c2 = Concept("b", np.zeros(4), tier=TIER_DOMAIN, project="p2")
    assert not can_interact(c1, c2)

def test_can_interact_origin_mismatch():
    c1 = Concept("a", np.zeros(4), tier=TIER_DOMAIN, origin="kb")
    c2 = Concept("b", np.zeros(4), tier=TIER_DOMAIN, origin="runtime")
    assert not can_interact(c1, c2)

def test_fusion_tier_protection(gravity):
    # Add two close concepts with different tiers
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    gravity.add_concept("c1", vec=vec, tier=TIER_DOMAIN)
    gravity.add_concept("c2", vec=vec, tier=TIER_SYSTEM)
    
    # Should not fuse
    fused = gravity.check_fusion_all()
    assert fused == 0
    assert "c1" in gravity.concepts
    assert "c2" in gravity.concepts

def test_fusion_valid(gravity):
    # Add two close concepts with same tier/project/origin
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    gravity.add_concept("c1", vec=vec, tier=TIER_DOMAIN)
    gravity.add_concept("c2", vec=vec, tier=TIER_DOMAIN)
    
    # Should fuse
    fused = gravity.check_fusion_all(base_threshold=0.5)
    assert fused == 1
    # One should be canonical, one alias
    assert (gravity.concepts["c1"].canonical_id == "c2" or 
            gravity.concepts["c2"].canonical_id == "c1")

def test_mitosis_tier_protection(gravity):
    # Add a massive system concept
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gravity.add_concept("sys:root", vec=vec, mass=10.0, tier=TIER_SYSTEM)
    
    # Attempt mitosis
    result = gravity.check_mitosis("sys:root")
    assert not result

def test_mitosis_mass_threshold(gravity):
    # Add a small domain concept
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gravity.add_concept("c1", vec=vec, mass=1.0, tier=TIER_DOMAIN)
    
    # Attempt mitosis (threshold is 2.0)
    result = gravity.check_mitosis("c1", mass_threshold=2.0)
    assert not result

def test_neighborhood_divergence(gravity):
    # Setup: c1 and c2 are close but have different neighbors
    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    gravity.add_concept("c1", vec=vec)
    gravity.add_concept("c2", vec=vec)
    
    # Use random normal vectors to avoid positive octant bias
    gravity.add_concept("n1", vec=np.random.randn(4))
    gravity.add_concept("n2", vec=np.random.randn(4))
    
    # Clear automatically created relations to ensure controlled test
    gravity.relations.clear()
    
    # c1 connected to n1
    k1 = tuple(sorted(("c1", "n1")))
    gravity.relations[k1] = 0.9
    
    # c2 connected to n2
    k2 = tuple(sorted(("c2", "n2")))
    gravity.relations[k2] = 0.9
    
    print(f"DEBUG: Relations: {gravity.relations.keys()}")
    
    div = gravity.neighborhood_divergence("c1", "c2")
    print(f"DEBUG: Div: {div}")
    assert div == 1.0  # Completely divergent
    
    # Connect both to n1
    k3 = tuple(sorted(("c2", "n1")))
    gravity.relations[k3] = 0.9
    # Now c1:{n1}, c2:{n1, n2} -> intersection={n1}, union={n1, n2} -> jaccard=0.5 -> div=0.5
    div = gravity.neighborhood_divergence("c1", "c2")
    assert div == 0.5
