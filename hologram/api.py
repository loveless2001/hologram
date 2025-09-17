from typing import Any, Optional
from dataclasses import dataclass

try:  # torch is only required when using CLIP encoders
    import torch  # type: ignore
except ImportError:  # pragma: no cover - triggered only when torch missing
    torch = None  # type: ignore

from .store import MemoryStore, Trace
from .glyphs import GlyphRegistry
from .config import VECTOR_DIM
from .embeddings import (
    ImageCLIP,
    TextCLIP,
    TextHasher,
    ImageStub,
    open_clip,
    get_clip_embed_dim,
)

@dataclass
class Hologram:
    store: MemoryStore
    glyphs: GlyphRegistry
    text_encoder: Any
    image_encoder: Any

    @classmethod
    def init(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        use_clip: bool = True,
    ):
        if use_clip:
            if torch is None or open_clip is None:
                raise RuntimeError(
                    "CLIP initialization requested but torch/open_clip are unavailable. "
                    "Install the dependencies or set use_clip=False for hashing fallback."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )

            embed_dim = get_clip_embed_dim(model)
            store = MemoryStore(vec_dim=embed_dim)

            text_enc = TextCLIP(model=model, device=device)
            img_enc = ImageCLIP(model=model, preprocess=preprocess, device=device)
        else:
            store = MemoryStore(vec_dim=VECTOR_DIM)
            text_enc = TextHasher(dim=VECTOR_DIM)
            img_enc = ImageStub(dim=VECTOR_DIM)

        return cls(
            store=store,
            glyphs=GlyphRegistry(store),
            text_encoder=text_enc,
            image_encoder=img_enc,
        )

    # --- Write ---
    def add_text(self, glyph_id: str, text: str, trace_id: Optional[str] = None, **meta):
        trace_id = trace_id or f"text:{abs(hash(text))%10**10}"
        vec = self.text_encoder.encode(text)
        tr = Trace(trace_id=trace_id, kind="text", content=text, vec=vec, meta=meta)
        self.glyphs.attach_trace(glyph_id, tr)
        return trace_id

    def add_image_path(self, glyph_id: str, path: str, trace_id: Optional[str] = None, **meta):
        trace_id = trace_id or f"image:{abs(hash(path))%10**10}"
        vec = self.image_encoder.encode_path(path)  # raises if file unreadable
        tr = Trace(trace_id=trace_id, kind="image", content=path, vec=vec, meta=meta)
        self.glyphs.attach_trace(glyph_id, tr)
        return trace_id

    # --- Read ---
    def recall_glyph(self, glyph_id: str):
        return self.glyphs.recall(glyph_id)

    def search_text(self, query: str, top_k: int = 5):
        qv = self.text_encoder.encode(query)
        return self.glyphs.search_across(qv, top_k=top_k)

    # Optional: image→image retrieval
    def search_image_path(self, path: str, top_k: int = 5):
        qv = self.image_encoder.encode_path(path)
        return self.glyphs.search_across(qv, top_k=top_k)

    # --- Utility ---
    def summarize_hit(self, trace: Trace, score: float) -> str:
        head = trace.content if trace.kind == "text" else f"{trace.kind}:{trace.content}"
        return f"[{trace.trace_id}] ({trace.kind}) score={score:.3f} :: {head[:80]}"

    # --- Persistence ---
    def save(self, path: str) -> None:
        self.store.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        use_clip: bool = True,
    ):
        store = MemoryStore.load(path)

        if use_clip:
            if torch is None or open_clip is None:
                raise RuntimeError(
                    "Cannot load CLIP-based hologram without torch/open_clip. "
                    "Install dependencies or load with use_clip=False."
                )
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            embed_dim = get_clip_embed_dim(model)
            if embed_dim != store.vec_dim:
                raise ValueError(
                    f"Loaded store has dimension {store.vec_dim}, "
                    f"but model '{model_name}' produces {embed_dim}."
                )
            text_enc = TextCLIP(model=model, device=device)
            img_enc = ImageCLIP(model=model, preprocess=preprocess, device=device)
        else:
            text_enc = TextHasher(dim=store.vec_dim)
            img_enc = ImageStub(dim=store.vec_dim)

        return cls(
            store=store,
            glyphs=GlyphRegistry(store),
            text_encoder=text_enc,
            image_encoder=img_enc,
        )
