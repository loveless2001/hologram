from typing import Optional
from dataclasses import dataclass
import torch

from .store import MemoryStore, Trace
from .glyphs import GlyphRegistry
from .embeddings import ImageCLIP, TextCLIP, open_clip, get_clip_embed_dim

@dataclass
class Hologram:
    store: MemoryStore
    glyphs: GlyphRegistry
    text_encoder: TextCLIP
    image_encoder: ImageCLIP

    @classmethod
    def init(cls, model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=device
        )

        embed_dim = get_clip_embed_dim(model)
        store = MemoryStore(vec_dim=embed_dim)

        text_enc = TextCLIP(model=model, device=device)
        img_enc  = ImageCLIP(model=model, preprocess=preprocess, device=device)

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
