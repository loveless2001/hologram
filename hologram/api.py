# hologram/api.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, List, Tuple
# from .smi import SymbolicMemoryInterface
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
    TextCLIP,
    TextHasher,
    TextMiniLM,
    ImageStub,
    open_clip,
    get_clip_embed_dim,
)
from .gravity import GravityField  # ← integrate Memory Gravity
from .text_utils import extract_concepts
from .manifold import LatentManifold
from .retrieval import extract_local_field
from .retrieval import extract_local_field
from .smi import MemoryPacket
from .mg_scorer import MGScore, mg_score


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
        encoder_mode: str = "default",  # "default", "hash", "minilm", "clip"
    ):
        # Encoder setup
        text_enc = None
        img_enc = None
        store_dim = VECTOR_DIM

        # 1. Resolve Text Encoder
        if encoder_mode == "minilm":
            text_enc = TextMiniLM()
            store_dim = 384  # MiniLM-L6-v2 dim
        elif encoder_mode == "hash":
            text_enc = TextHasher(dim=VECTOR_DIM)
            store_dim = VECTOR_DIM
        elif encoder_mode == "clip" or (encoder_mode == "default" and use_clip):
            if torch is None or open_clip is None:
                raise RuntimeError("CLIP requested but dependencies missing.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            store_dim = get_clip_embed_dim(model)
            text_enc = TextCLIP(model=model, device=device)
            img_enc = ImageCLIP(model=model, preprocess=preprocess, device=device)
        else:
            # Fallback default -> hash if use_clip=False
            text_enc = TextHasher(dim=VECTOR_DIM)
            store_dim = VECTOR_DIM

        # 2. Resolve Image Encoder (if not already set by CLIP)
        if img_enc is None:
            img_enc = ImageStub(dim=store_dim)

        store = MemoryStore(vec_dim=store_dim)

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

    def retrieve(self, query: str) -> MemoryPacket:
        """
        Perform a dynamic retrieval using probe physics.
        Returns a structured Memory Packet (SMI).
        """
        # 1. Encode query via Manifold
        qv = self.manifold.align_text(query, self.text_encoder)
        
        # 2. Initialize Packet
        if not self.field:
            # Fallback if gravity is disabled
            return MemoryPacket(seed=query, nodes=[], edges=[], glyphs=[])
            
        # 3. Spawn Probe & Simulate Trajectory
        probe = self.field.spawn_probe(qv)
        self.field.simulate_trajectory(probe, steps=5)
        
        # 4. Extract Local Field
        data = extract_local_field(self.field, probe)
        
        # 5. Wrap in Memory Packet
        packet = MemoryPacket(
            seed=query,
            nodes=data["nodes"],
            edges=data["edges"],
            glyphs=data["glyphs"],
            trajectory_steps=data["trajectory_steps"]
        )
        
        return packet

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

    # --- MG Scoring ---
    def score_text(self, texts: List[str]) -> MGScore:
        """
        Compute MGScore for a list of text strings.
        Useful for evaluating coherence of a generated paragraph or a set of ideas.
        """
        vectors = [self.manifold.align_text(t, self.text_encoder) for t in texts]
        return mg_score(vectors)

    def score_trace(self, trace_ids: List[str]) -> MGScore:
        """
        Compute MGScore for a list of trace IDs.
        Useful for evaluating the coherence of a retrieval result or a memory cluster.
        """
        vectors = []
        for tid in trace_ids:
            # Try to find in field first (faster?) or store
            # Store is source of truth for vectors
            # But field might have updated positions if gravity is on?
            # Actually store.traces has the original vector. 
            # Gravity field has the *current* position.
            # We should probably use the *current* position if gravity is active.
            
            if self.field and tid in self.field.sim.concepts:
                # If it's a concept in gravity
                vectors.append(self.field.sim.concepts[tid].vec)
            elif tid in self.store.traces:
                # Fallback to static trace vector
                vectors.append(self.store.traces[tid].vec)
            else:
                # Skip unknown or warn?
                continue
        
        if not vectors:
            # Return a zero-score or raise?
            # Let's return a dummy score with 0 coherence
            return MGScore(0.0, 0.0, 0.0, 0.0, np.zeros(self.store.vec_dim))
            
        return mg_score(vectors)

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
        encoder_mode: str = "default",
    ):
        store = MemoryStore.load(path)

        # Encoder setup logic (duplicated from init, should refactor but inline for now)
        text_enc = None
        img_enc = None
        
        # 1. Resolve Text Encoder
        if encoder_mode == "minilm":
            text_enc = TextMiniLM()
        elif encoder_mode == "hash":
            text_enc = TextHasher(dim=store.vec_dim)
        elif encoder_mode == "clip" or (encoder_mode == "default" and use_clip):
            if torch is None or open_clip is None:
                raise RuntimeError("CLIP requested but dependencies missing.")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device
            )
            embed_dim = get_clip_embed_dim(model)
            if embed_dim != store.vec_dim:
                 # Warn but allow if user knows what they are doing? 
                 # Or maybe store.vec_dim is 384 (MiniLM) and we try to load CLIP (512).
                 # This will crash later.
                 pass
            text_enc = TextCLIP(model=model, device=device)
            img_enc = ImageCLIP(model=model, preprocess=preprocess, device=device)
        else:
            # Fallback default -> hash
            text_enc = TextHasher(dim=store.vec_dim)

        # 2. Resolve Image Encoder
        if img_enc is None:
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
    # def init_memory(self, save_path="data/smi_state.json"):
    #     self.smi = SymbolicMemoryInterface(self.store, self.glyphs, save_path=save_path)
    #     return self.smi