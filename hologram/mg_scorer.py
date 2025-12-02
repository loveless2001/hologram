"""
hologram.mg_scorer
==================

Memory Gravity (MG) scoring utilities.

This module provides a first-pass implementation of the MG Scorer, which
computes geometric signatures over sets/sequences of semantic vectors:

- coherence       : how tightly clustered the vectors are
- curvature       : how non-linear the sequence of vectors is
- semantic_entropy: how many principal directions of variation exist
- collapse_risk   : combined instability indicator
- gradient        : direction of semantic tension relative to centroid

Intended to be used on:
- concept trace vectors
- glyph trace vectors
- sentence/paragraph/document embeddings
- nodes in a MemoryPacket from Hologram's SMI layer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Optional

import numpy as np


ArrayLike = np.ndarray


@dataclass
class MGScore:
    """
    Container for Memory Gravity (MG) scores.

    Attributes
    ----------
    coherence : float
        In [0, 1]. Higher means the vectors form a tight, coherent cluster.
    curvature : float
        Dimensionless proxy for semantic curvature. Values near 1.0 indicate
        almost-linear transitions; significantly lower values indicate stronger
        "bending" in the sequence of vectors.
    entropy : float
        Semantic entropy based on eigenvalues of the covariance matrix.
        Higher values indicate more dispersed, multi-directional variation.
    collapse_risk : float
        Composite risk indicator derived from coherence, entropy, and curvature.
        Higher means the configuration is more likely to be unstable or collapsing.
    gradient : np.ndarray
        Average deviation from the centroid. This can be interpreted as the
        direction of semantic tension in the latent manifold.
    """

    coherence: float
    curvature: float
    entropy: float
    collapse_risk: float
    gradient: ArrayLike


def _to_matrix(vectors: Iterable[ArrayLike]) -> ArrayLike:
    """
    Convert an iterable of vectors into a 2D NumPy array.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        Iterable of 1D vectors with the same dimensionality.

    Returns
    -------
    np.ndarray
        2D array of shape (n_vectors, dim).

    Raises
    ------
    ValueError
        If no vectors are provided.
    """
    arr = np.array(list(vectors), dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a collection of 1D vectors; got shape "
                         f"{arr.shape!r}")
    if arr.shape[0] == 0:
        raise ValueError("Cannot compute MG score on empty vector set.")
    return arr


def coherence(vectors: Iterable[ArrayLike]) -> float:
    """
    Compute coherence score for a set of vectors.

    Coherence is implemented as 1 - average cosine distance. It reflects
    how tightly clustered the vectors are in the latent space.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        Collection of 1D vectors.

    Returns
    -------
    float
        Coherence in [0, 1]. Higher means more coherent.

    Notes
    -----
    - If there is only one vector, returns 1.0 by convention.
    """
    mat = _to_matrix(vectors)
    n = mat.shape[0]
    if n == 1:
        return 1.0

    # Normalize for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    mat_norm = mat / norms

    sims = mat_norm @ mat_norm.T  # (n, n)
    # cosine distance = 1 - cosine similarity
    dists = 1.0 - sims

    # exclude diagonal (distance with itself)
    mask = ~np.eye(n, dtype=bool)
    avg_dist = float(dists[mask].mean())
    # Coherence = 1 - average distance
    coh = max(0.0, min(1.0, 1.0 - avg_dist))
    return coh


def curvature(vectors: Sequence[ArrayLike]) -> float:
    """
    Estimate semantic curvature (straightness) of an ordered sequence.

    Uses the 'Straightness' index for triplets:
        kappa = ||v_{i+2} - v_i|| / (||v_{i+1} - v_i|| + ||v_{i+2} - v_{i+1}||)

    Interpretation:
    - 1.0 : Perfectly linear (v0 -> v1 -> v2 is a straight line).
    - < 1.0 : Curved path.
    - ~ 0.7 : 90 degree turn.
    - 0.0 : 180 degree turn (backtracking).

    Parameters
    ----------
    vectors : Sequence[np.ndarray]
        Ordered sequence of 1D vectors.

    Returns
    -------
    float
        Average straightness index.
    """
    mat = _to_matrix(vectors)
    n = mat.shape[0]
    if n < 3:
        return 1.0

    local_vals: list[float] = []
    for i in range(n - 2):
        v0 = mat[i]
        v1 = mat[i + 1]
        v2 = mat[i + 2]
        
        dist_total = np.linalg.norm(v1 - v0) + np.linalg.norm(v2 - v1)
        dist_net = np.linalg.norm(v2 - v0)
        
        if dist_total <= 1e-12:
            # No movement, consider it linear/stable
            local_vals.append(1.0)
        else:
            local_vals.append(dist_net / dist_total)

    if not local_vals:
        return 1.0

    return float(np.mean(local_vals))


def semantic_entropy(vectors: Iterable[ArrayLike]) -> float:
    """
    Compute semantic entropy over a set of vectors.

    Entropy is defined using the eigenvalues of the covariance matrix
    of the vectors:

        1. Compute covariance matrix Σ.
        2. Compute eigenvalues λ_i.
        3. Normalize λ_i to sum to 1.
        4. Compute H = -Σ λ_i log(λ_i).

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        Collection of 1D vectors.

    Returns
    -------
    float
        Semantic entropy. Higher values indicate more dispersed,
        multi-directional variation in the set.

    Notes
    -----
    - Eigenvalues are clamped to a small positive value to avoid log(0).
    """
    mat = _to_matrix(vectors)
    n = mat.shape[0]
    if n < 2:
        return 0.0

    # Shape: (dim, n) expected by np.cov if rowvar=True; we want rowvar=False
    cov = np.cov(mat, rowvar=False)
    # Symmetrize just in case of numerical noise
    cov = 0.5 * (cov + cov.T)

    vals, _ = np.linalg.eigh(cov)
    # Clamp to small positive to avoid log(0)
    vals = np.maximum(vals, 1e-12)
    total = float(vals.sum())
    if total <= 0:
        return 0.0

    probs = vals / total
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy


def collapse_risk(
    coherence_score: float,
    entropy_score: float,
    curvature_score: float,
    *,
    entropy_scale: Optional[float] = None,
) -> float:
    """
    Compute a composite collapse risk score.

    A simple heuristic combination:

        risk_raw = (1 - coherence) * scaled_entropy * (1 - curvature)

    where:
    - coherence in [0, 1]: low coherence contributes to higher risk
    - curvature ~1.0: near-linear; lower curvature contributes to higher risk
    - entropy: scaled to a reasonable range before combination

    Parameters
    ----------
    coherence_score : float
        Coherence in [0, 1].
    entropy_score : float
        Semantic entropy (unbounded positive).
    curvature_score : float
        Curvature proxy, typically around 1.0 for linear-ish behavior.
    entropy_scale : float, optional
        If provided, entropy_score is divided by this value to normalize it
        before combining. If None, a soft normalization based on
        (1 + entropy_score) is used.

    Returns
    -------
    float
        Collapse risk in [0, +∞). You may want to rescale or clip it for
        downstream usage.
    """
    coh = float(coherence_score)
    curv = float(curvature_score)
    ent = float(entropy_score)

    if entropy_scale is not None and entropy_scale > 0:
        ent_norm = ent / entropy_scale
    else:
        # Soft normalization: log-like compression
        ent_norm = np.log1p(ent)  # log(1 + ent)

    risk_raw = (1.0 - coh) * ent_norm * (1.0 - min(curv, 1.0))
    # Ensure non-negative
    return float(max(0.0, risk_raw))


def gradient(vectors: Iterable[ArrayLike]) -> ArrayLike:
    """
    Compute the average deviation from the centroid (semantic gradient).

    This is a coarse proxy for the 'direction of semantic tension' within
    the set: which direction in the latent space the configuration is
    biased towards.

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        Collection of 1D vectors.

    Returns
    -------
    np.ndarray
        1D vector of the same dimensionality as the inputs.
    """
    mat = _to_matrix(vectors)
    centroid = mat.mean(axis=0)
    deviations = mat - centroid
    grad = deviations.mean(axis=0)
    return grad


def mg_score(vectors: Iterable[ArrayLike]) -> MGScore:
    """
    Compute the full MGScore for a collection or sequence of vectors.

    This is the main entry point for the v0.1 MG Scorer. It aggregates:
    - coherence
    - curvature
    - semantic entropy
    - collapse risk (derived)
    - gradient

    Parameters
    ----------
    vectors : Iterable[np.ndarray]
        Collection (unordered) or sequence (ordered) of 1D vectors.

    Returns
    -------
    MGScore
        Dataclass instance with all metrics populated.
    """
    mat = _to_matrix(vectors)

    coh = coherence(mat)
    curv = curvature(mat)
    ent = semantic_entropy(mat)
    risk = collapse_risk(coh, ent, curv)
    grad = gradient(mat)

    return MGScore(
        coherence=coh,
        curvature=curv,
        entropy=ent,
        collapse_risk=risk,
        gradient=grad,
    )


if __name__ == "__main__":
    # Minimal smoke test / example usage
    rng = np.random.default_rng(42)

    # Simulate a small coherent cluster in 5D
    base = rng.normal(size=5)
    samples = base + 0.05 * rng.normal(size=(10, 5))

    score = mg_score(samples)
    print("MGScore demo:")
    print(f"  coherence    : {score.coherence:.4f}")
    print(f"  curvature    : {score.curvature:.4f}")
    print(f"  entropy      : {score.entropy:.4f}")
    print(f"  collapse_risk: {score.collapse_risk:.4f}")
    print(f"  gradient norm: {np.linalg.norm(score.gradient):.4f}")
