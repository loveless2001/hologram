from typing import List, Optional
import hashlib
import numpy as np
from .config import VECTOR_DIM, SEED

import PIL.Image as Image
import torch
import open_clip  # re-exported at bottom


# -------------------------
# Lightweight hashing encoders (keep for demos)
# -------------------------
class TextHasher:
    def __init__(self, dim: int = VECTOR_DIM, seed: int = SEED):
        self.dim = dim
        self.seed = seed

    def _hash(self, s: str) -> int:
        h = hashlib.blake2b((str(self.seed) + s).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little")

    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().split():
            idx = self._hash(tok) % self.dim
            sign = 1 if (self._hash(tok + "sign") % 2 == 0) else -1
            vec[idx] += sign * 1.0
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        return np.stack([self.encode(t) for t in texts], axis=0)


class ImageStub:
    def __init__(self, dim: int = VECTOR_DIM, seed: int = SEED):
        self.dim = dim
        self.seed = seed

    def _hash(self, b: bytes) -> int:
        h = hashlib.blake2b(b + bytes([self.seed]), digest_size=8).digest()
        return int.from_bytes(h, "little")

    def encode_path(self, path: str) -> np.ndarray:
        return self.encode_bytes(path.encode("utf-8"))

    def encode_bytes(self, data: bytes) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        step = max(1, len(data) // 16)
        for i in range(0, len(data), step):
            chunk = data[i : i + step]
            idx = self._hash(chunk) % self.dim
            sign = 1 if (self._hash(chunk + b"sign") % 2 == 0) else -1
            vec[idx] += sign * 1.0
        norm = np.linalg.norm(vec) + 1e-8
        return vec / norm


# -------------------------
# CLIP encoders (share one model across text + image)
# -------------------------
class ImageCLIP:
    """
    Image encoder using OpenCLIP. If a model/preprocess is provided, reuse it
    so vectors align with TextCLIP from the same tower.
    """
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        preprocess: Optional[torch.nn.Module] = None,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None or preprocess is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
        else:
            self.model = model
            self.preprocess = preprocess
        self.model.eval()

    @torch.no_grad()
    def encode_path(self, path: str) -> np.ndarray:
        # use context manager to avoid file handle leaks
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        feats = self.model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")


class TextCLIP:
    """
    Text encoder using the SAME OpenCLIP model as ImageCLIP.
    """
    def __init__(self, model: torch.nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        toks = open_clip.tokenize([text]).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")


# Utility: get embedding dim from a CLIP model (works across variants)
def get_clip_embed_dim(model: torch.nn.Module) -> int:
    if hasattr(model, "text_projection") and model.text_projection is not None:
        return int(model.text_projection.shape[-1])
    if hasattr(model, "visual") and hasattr(model.visual, "output_dim"):
        return int(model.visual.output_dim)
    raise RuntimeError("Could not infer embed_dim from CLIP model.")


# Re-export for convenient import in api.py
__all__ = [
    "TextHasher",
    "ImageStub",
    "ImageCLIP",
    "TextCLIP",
    "get_clip_embed_dim",
    "open_clip",
]
