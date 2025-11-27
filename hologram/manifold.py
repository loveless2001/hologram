# hologram/manifold.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Union
import numpy as np

@dataclass
class LatentManifold:
    """
    Unified latent manifold for projecting disparate vector sources into a
    consistent physical space for the GravityField.
    
    Responsibilities:
    1. Normalization: Ensure all vectors are unit length.
    2. Alignment: (Future) Align different embedding spaces.
    3. Consistency: Act as the single source of truth for vector entry.
    """
    dim: int
    
    def project(self, vec: np.ndarray) -> np.ndarray:
        """
        Project a raw vector onto the latent manifold.
        Currently performs L2 normalization.
        """
        vec = vec.astype("float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec
    
    def align_text(self, text: str, encoder: Any) -> np.ndarray:
        """
        Encode text using the provided encoder and project onto the manifold.
        """
        raw_vec = encoder.encode(text)
        return self.project(raw_vec)
    
    def align_image(self, image_path: str, encoder: Any) -> np.ndarray:
        """
        Encode image path using the provided encoder and project onto the manifold.
        """
        raw_vec = encoder.encode_path(image_path)
        return self.project(raw_vec)
