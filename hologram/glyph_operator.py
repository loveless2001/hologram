# hologram/glyph_operator.py
"""
Glyph-conditioned transform operator.

Doc spec (north-star): T_g(z) = P_k R_g z
  - R_g = glyph-specific orthogonal rotation
  - P_k = top-k dimension projection

Phase 1: identity transforms (proves routing thesis)
Phase 2+: real R_g + P_k swap without rewriting retrieval path
"""
import numpy as np
from typing import Optional


class GlyphOperator:
    """
    Per-glyph transform operator for glyph-conditioned retrieval.

    Each glyph defines a retrieval operator that transforms vectors
    into a glyph-specific subspace. In Phase 1 this is identity;
    Phase 2+ introduces orthogonal rotation + dimension projection.
    """

    def __init__(self, glyph_id: str, dim: int):
        self.glyph_id = glyph_id
        self.dim = dim
        # Phase 2+ will add:
        # self._rotation: np.ndarray  # R_g orthogonal matrix (dim x dim)
        # self._projection_k: int     # P_k top-k dimensions
        # self._scale_mask: Optional[np.ndarray]  # D_g scaling mask
        # self._phase_sign: Optional[np.ndarray]  # S_g polarity

    def transform_query(self, vec: np.ndarray) -> np.ndarray:
        """Transform query vector into this glyph's subspace.

        Phase 1: identity (returns vec unchanged).
        Phase 2+: P_k @ R_g @ vec
        """
        return vec

    def transform_trace(self, vec: np.ndarray) -> np.ndarray:
        """Transform trace vector for storage in this glyph's shard index.

        Phase 1: identity (returns vec unchanged).
        Phase 2+: P_k @ R_g @ vec
        """
        return vec

    @property
    def output_dim(self) -> int:
        """Dimension of vectors after transform.

        Phase 1: same as input dim.
        Phase 2+: k (projected subspace dimension).
        """
        return self.dim
