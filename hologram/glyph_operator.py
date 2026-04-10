# hologram/glyph_operator.py
"""
Glyph-conditioned transform operator.

Doc spec: T_g(z) = P_k R_g z
  - R_g = glyph-specific orthonormal basis (dim x dim or dim x k)
  - P_k = top-k dimension projection

Supports three modes:
  - identity: pass-through (use_projection=False)
  - random: fixed random orthogonal R_g per glyph (default)
  - learned: PCA-derived basis from glyph's trace data (via learn_from_traces)
"""
import hashlib
import numpy as np
from typing import Optional, List


def _random_orthogonal(dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic random orthogonal matrix via QR decomposition.

    Same seed + dim always produces the same rotation, ensuring
    reproducibility across shard rebuilds.
    """
    rng = np.random.RandomState(seed)
    A = rng.randn(dim, dim).astype("float32")
    Q, R = np.linalg.qr(A)
    Q *= np.sign(np.diag(R))
    return Q.astype("float32")


class GlyphOperator:
    """
    Per-glyph transform operator for glyph-conditioned retrieval.

    Each glyph defines a retrieval operator T_g(z) = P_k @ R_g @ z
    that projects vectors into a glyph-specific subspace. Supports
    random orthogonal R_g (default) and PCA-learned R_g from trace data.

    Args:
        glyph_id: Unique glyph identifier (used to derive rotation seed)
        dim: Input embedding dimension
        k: Output subspace dimension (default: dim // 8, min 8)
        use_projection: If True, apply R_g + P_k. If False, pass-through
    """

    def __init__(self, glyph_id: str, dim: int, k: Optional[int] = None,
                 use_projection: bool = True):
        self.glyph_id = glyph_id
        self.dim = dim
        self.use_projection = use_projection
        self._learned = False

        if use_projection:
            digest = hashlib.blake2b(glyph_id.encode("utf-8"), digest_size=4).digest()
            self._seed = int.from_bytes(digest, "little") % (2**31)
            self._k = k if k is not None else max(dim // 8, 8)
            # Start with random rotation; learn_from_traces() replaces it
            self._basis = _random_orthogonal(dim, self._seed)[:self._k]
            # _basis shape: (k, dim) — each row is a basis vector
        else:
            self._basis = None
            self._k = dim

    def learn_from_traces(self, vecs: List[np.ndarray],
                          min_traces: int = 5) -> bool:
        """Learn glyph-specific basis from trace vectors via PCA/SVD.

        Computes principal components of the glyph's trace data and uses
        them as the projection basis. Falls back to random R_g if too
        few traces are available.

        Args:
            vecs: List of trace vectors belonging to this glyph
            min_traces: Minimum traces needed for stable PCA

        Returns:
            True if PCA basis was learned, False if fell back to random
        """
        if not self.use_projection:
            return False
        if len(vecs) < min_traces:
            self._learned = False
            return False

        # Stack and center trace vectors
        mat = np.stack(vecs).astype("float32")
        mean = mat.mean(axis=0, keepdims=True)
        centered = mat - mean

        # SVD to get principal components (orthonormal by construction)
        # U @ diag(S) @ Vt = centered; rows of Vt are principal directions
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)

        # Take top-k components as the learned basis
        k = min(self._k, len(S))
        self._basis = Vt[:k].astype("float32")
        self._k = k
        self._learned = True
        return True

    def set_basis(self, basis: np.ndarray) -> None:
        """Set an externally-computed basis (e.g., shared discriminant basis).

        Args:
            basis: (k, dim) orthonormal matrix — each row is a basis vector
        """
        self._basis = basis.astype("float32")
        self._k = basis.shape[0]
        self._learned = True

    def transform(self, vec: np.ndarray) -> np.ndarray:
        """Apply T_g(z) = basis @ z. Projects into glyph's k-dim subspace."""
        if not self.use_projection:
            return vec
        # _basis is (k, dim), vec is (dim,) → result is (k,)
        return self._basis @ vec.astype("float32")

    def transform_query(self, vec: np.ndarray) -> np.ndarray:
        """Transform query vector into this glyph's subspace."""
        return self.transform(vec)

    def transform_trace(self, vec: np.ndarray) -> np.ndarray:
        """Transform trace vector for storage in this glyph's shard index."""
        return self.transform(vec)

    @property
    def output_dim(self) -> int:
        """Dimension of vectors after transform."""
        return self._k if self.use_projection else self.dim

    @property
    def is_learned(self) -> bool:
        """Whether the basis was learned from data (vs random)."""
        return self._learned
