import pytest
import numpy as np
from hologram.mg_scorer import mg_score, coherence, curvature, semantic_entropy, collapse_risk, gradient

def test_coherence_perfect():
    # Identical vectors should have coherence 1.0
    v = np.array([1.0, 0.0, 0.0])
    vectors = [v, v, v]
    assert coherence(vectors) >= 0.999

def test_curvature_linear():
    # Perfectly linear sequence: v0 -> v1 -> v2
    # v1-v0 = [1,0], v2-v1 = [1,0].
    # dist_total = 1 + 1 = 2.
    # dist_net = ||v2-v0|| = 2.
    # Ratio = 2/2 = 1.0.
    
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([2.0, 0.0])
    assert curvature([v0, v1, v2]) == 1.0

def test_curvature_turn():
    # 90 deg turn
    # v0=[0,0], v1=[1,0], v2=[1,1]
    # dist_total = 1 + 1 = 2.
    # dist_net = ||[1,1]|| = 1.414.
    # Ratio = 0.707.
    
    v0 = np.array([0.0, 0.0])
    v1 = np.array([1.0, 0.0])
    v2 = np.array([1.0, 1.0])
    curv = curvature([v0, v1, v2])
    assert 0.70 < curv < 0.711

def test_collapse_risk():
    # High risk: low coherence, high entropy, low curvature
    # Formula: (1 - coh) * log1p(ent) * (1 - curv)
    # coh=0.1, ent=2.0, curv=0.5
    # Risk = 0.9 * log(3.0) * 0.5
    # log(3) approx 1.098
    # Risk approx 0.9 * 1.098 * 0.5 = 0.494
    
    risk = collapse_risk(0.1, 2.0, 0.5)
    assert risk > 0.45
    
    # Low risk: high coherence
    risk_safe = collapse_risk(0.95, 0.1, 0.5)
    # (0.05) * log(1.1) * 0.5 approx 0.05 * 0.1 * 0.5 = small
    assert risk_safe < 0.1
