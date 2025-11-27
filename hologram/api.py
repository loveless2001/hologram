# hologram/api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
from .smi import SymbolicMemoryInterface
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

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
from .gravity import GravityField  # ← integrate Memory Gravity
from .text_utils import extract_concepts
from .manifold import LatentManifold


@dataclass
class Hologram:
    """
    Unified holographic memory system:
    - encodes text/images into vectors
    - anchors them in a gravitational vector field
    - manages symbolic traces via GlyphRegistry
    """
    store: MemoryStore
    glyphs: GlyphRegistry
    text_encoder: Any
    image_encoder: Any
    manifold: LatentManifold
    field: Optional[GravityField] = None

    # --- Initialization ---
    @classmethod
    def init(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        use_clip: bool = True,
        use_gravity: bool = True,
    ):
        # Encoder setup
        if use_clip:
            if torch is None or open_clip is None:
                raise RuntimeError(
                    "CLIP initialization requested but torch/open_clip are unavailable. "
                    "Install dependencies or set use_clip=False for hashing fallback."
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

        manifold = LatentManifold(dim=store.vec_dim)
        field = GravityField(dim=store.vec_dim) if use_gravity else None

        return cls(
            store=store,
            glyphs=GlyphRegistry(store),
            text_encoder=text_enc,
            image_encoder=img_enc,
            manifold=manifold,
            field=field,
        )

    # --- Write operations ---
    def add_text(self, glyph_id: str, text: str, trace_id: Optional[str] = None, do_extract_concepts: bool = False, **meta):
        trace_id = trace_id or f"text:{abs(hash(text))%10**10}"
        # Use manifold for alignment
        vec = self.manifold.align_text(text, self.text_encoder)
        
        tr = Trace(trace_id=trace_id, kind="text", content=text, vec=vec, meta=meta)
        self.glyphs.attach_trace(glyph_id, tr)
        if self.field:
            self.field.add(trace_id, vec)
            
            if do_extract_concepts:
                concepts = extract_concepts(text)
                for concept_text in concepts:
                    # Concepts also go through manifold
                    c_vec = self.manifold.align_text(concept_text, self.text_encoder)
                    self.field.add(concept_text, vec=c_vec)
                    if self.field.check_mitosis(concept_text):
                        print(f"[Gravity] Mitosis occurred for '{concept_text}'")
                        
        return trace_id

    def add_image_path(self, glyph_id: str, path: str, trace_id: Optional[str] = None, **meta):
        trace_id = trace_id or f"image:{abs(hash(path))%10**10}"
        # Use manifold for alignment
        vec = self.manifold.align_image(path, self.image_encoder)
        
        tr = Trace(trace_id=trace_id, kind="image", content=path, vec=vec, meta=meta)
        self.glyphs.attach_trace(glyph_id, tr)
        if self.field:
            self.field.add(trace_id, vec)
        return trace_id

    # --- Read operations ---
    def recall_glyph(self, glyph_id: str):
        return self.glyphs.recall(glyph_id)

    def search_text(self, query: str, top_k: int = 5):
        # Use manifold for alignment
        qv = self.manifold.align_text(query, self.text_encoder)
        return self.glyphs.search_across(qv, top_k=top_k)

    def search_image_path(self, path: str, top_k: int = 5):
        # Use manifold for alignment
        qv = self.manifold.align_image(path, self.image_encoder)
        return self.glyphs.search_across(qv, top_k=top_k)

    # --- Field analytics ---
    def field_state(self):
        """Return current gravitational field projection if active."""
        if not self.field:
            return None
        return self.field.project2d()

    def decay(self, steps: int = 1):
        """Advance field decay (Memory Gravity loss)."""
        if self.field:
            self.field.step_decay(steps=steps)

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
        use_gravity: bool = True,
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

        # Use the gravity state from the store if available, otherwise create new
        if use_gravity and hasattr(store, 'sim') and store.sim is not None:
            # Wrap the existing Gravity instance in a GravityField
            field = GravityField(dim=store.vec_dim)
            field.sim = store.sim  # Use the restored gravity state
        elif use_gravity:
            field = GravityField(dim=store.vec_dim)
        else:
            field = None
            
        manifold = LatentManifold(dim=store.vec_dim)

        return cls(
            store=store,
            glyphs=GlyphRegistry(store),
            text_encoder=text_enc,
            image_encoder=img_enc,
            manifold=manifold,
            field=field,
        )
    def init_memory(self, save_path="data/smi_state.json"):
        self.smi = SymbolicMemoryInterface(self.store, self.glyphs, save_path=save_path)
        return self.smi