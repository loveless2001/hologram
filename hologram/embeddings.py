from typing import List, Optional, Any
import hashlib
import numpy as np
from .config import VECTOR_DIM, SEED

import PIL.Image as Image

try:  # Optional heavy deps; fallback encoders do not require them
    import torch  # type: ignore
except ImportError:  # pragma: no cover - exercised only when torch missing
    torch = None  # type: ignore

try:
    import open_clip  # type: ignore  # re-exported at bottom
except ImportError:  # pragma: no cover - exercised only when open_clip missing
    open_clip = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None



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
        model: Optional[Any] = None,
        preprocess: Optional[Any] = None,
        model_name: str = None,
        pretrained: str = None,
        device: Optional[str] = None,
    ):
        from .config import Config
        model_name = model_name or Config.embedding.CLIP_MODEL
        pretrained = pretrained or Config.embedding.CLIP_PRETRAINED
        device = device or Config.embedding.DEVICE
        if torch is None or open_clip is None:
            raise RuntimeError(
                "open_clip and torch are required for CLIP-based image encoding. "
                "Install open-clip-torch and torch, or use the hashing fallback via"
                " Hologram.init(use_clip=False)."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if model is None or preprocess is None:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device
            )
        else:
            self.model = model
            self.preprocess = preprocess
        self.model.eval()

    def encode_path(self, path: str) -> np.ndarray:
        if torch is None:
            raise RuntimeError(
                "torch is required for CLIP-based image encoding. Install torch or "
                "use Hologram.init(use_clip=False)."
            )
        # use context manager to avoid file handle leaks
        with Image.open(path) as img:
            img = img.convert("RGB")
            with torch.no_grad():
                tensor = self.preprocess(img).unsqueeze(0).to(self.device)
                feats = self.model.encode_image(tensor)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")


class TextCLIP:
    """
    Text encoder using the SAME OpenCLIP model as ImageCLIP.
    """
    def __init__(self, model: Any, device: Optional[str] = None):
        from .config import Config
        device = device or Config.embedding.DEVICE
        if torch is None:
            raise RuntimeError(
                "torch is required for CLIP-based text encoding. Install torch or "
                "use the hashing fallback via Hologram.init(use_clip=False)."
            )

        self.model = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        if torch is None or open_clip is None:
            raise RuntimeError(
                "open_clip and torch are required for CLIP tokenization. Install the"
                " dependencies or use Hologram.init(use_clip=False)."
            )
        with torch.no_grad():
            toks = open_clip.tokenize([text]).to(self.device)
            feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.squeeze(0).detach().cpu().numpy().astype("float32")


class TextMiniLM:
    def __init__(self, model_name: str = None):
        from .config import Config
        model_name = model_name or Config.embedding.MINILM_MODEL
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is required for MiniLM encoding. "
                "Install it with `pip install sentence-transformers`."
            )
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.astype("float32")


# Utility: get embedding dim from a CLIP model (works across variants)
def get_clip_embed_dim(model: Any) -> int:
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
    "TextMiniLM",
    "get_clip_embed_dim",
    "open_clip",
]
