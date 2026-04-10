# hologram/glyph_operator.py
"""
Glyph-conditioned transform operator.

Doc spec: T_g(z) = P_k R_g z
  - R_g = glyph-specific orthogonal rotation matrix (dim x dim)
  - P_k = top-k dimension projection (selects first k dims after rotation)

V1 (current): fixed random orthogonal R_g per glyph, same k for all glyphs.
Future: learned R_g from data, per-glyph k, D_g scaling mask, S_g phase sign.
"""
import hashlib
import numpy as np
from typing import Optional


def _random_orthogonal(dim: int, seed: int) -> np.ndarray:
    """Generate a deterministic random orthogonal matrix via QR decomposition.

    Same seed + dim always produces the same rotation, ensuring
    reproducibility across shard rebuilds.
    """
    rng = np.random.RandomState(seed)
    # Random matrix -> QR decomposition -> Q is orthogonal
    A = rng.randn(dim, dim).astype("float32")
    Q, R = np.linalg.qr(A)
    # Fix sign ambiguity: ensure det(Q) > 0 (proper rotation)
    Q *= np.sign(np.diag(R))
    return Q.astype("float32")


class GlyphOperator:
    """
    Per-glyph transform operator for glyph-conditioned retrieval.

    Each glyph defines a retrieval operator T_g(z) = P_k @ R_g @ z
    that rotates vectors into a glyph-specific basis and projects
    into a k-dimensional subspace. Different glyphs expose different
    aspects of the same embedding via distinct rotations.

    Args:
        glyph_id: Unique glyph identifier (used to derive rotation seed)
        dim: Input embedding dimension
        k: Output subspace dimension (default: dim // 8, min 8)
        use_projection: If True, apply R_g + P_k. If False, pass-through (no transform)
    """

    def __init__(self, glyph_id: str, dim: int, k: Optional[int] = None,
                 use_projection: bool = True):
        self.glyph_id = glyph_id
        self.dim = dim
        self.use_projection = use_projection

        if use_projection:
            # Derive process-stable seed from glyph_id via blake2b digest
            digest = hashlib.blake2b(glyph_id.encode("utf-8"), digest_size=4).digest()
            self._seed = int.from_bytes(digest, "little") % (2**31)
            self._rotation = _random_orthogonal(dim, self._seed)
            # Default k: dim // 8 (e.g., 384 -> 48, 128 -> 16), minimum 8
            self._k = k if k is not None else max(dim // 8, 8)
            # P_k is implicit: just take first _k dims after rotation
        else:
            self._rotation = None
            self._k = dim

    def transform(self, vec: np.ndarray) -> np.ndarray:
        """Apply T_g(z) = P_k @ R_g @ z. Shared by query and trace transforms."""
        if not self.use_projection:
            return vec
        rotated = self._rotation @ vec.astype("float32")
        return rotated[:self._k]

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
