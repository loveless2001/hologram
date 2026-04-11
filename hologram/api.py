# hologram/api.py
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
from pathlib import Path
import numpy as np
import faiss


def _stable_id(prefix: str, text: str) -> str:
    """Generate a process-stable ID from text using blake2b digest."""
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()
    return f"{prefix}:{digest}"

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from .store import MemoryStore, Trace
from .glyphs import GlyphRegistry
from .chunking import chunk_text, Chunk
from .config import VECTOR_DIM
from .embeddings import (
    ImageCLIP,
    TextCLIP,
    TextHasher,
    TextMiniLM,
    ImageStub,
    open_clip,
    get_clip_embed_dim,
)
from .gravity import GravityField
from .text_utils import extract_concepts, normalize_text
from .manifold import LatentManifold
from .retrieval import extract_local_field
from .smi import MemoryPacket
from .mg_scorer import MGScore, mg_score
from .glyph_router import GlyphRouter
from .parsers import ParsedDocument, parse_pdf


class _GlobalPCAIndex:
    """Lazy global PCA index used by dynamic retrieval policy."""

    def __init__(self, store: MemoryStore, pca_dim: int):
        traces = [
            trace for trace in store.traces.values()
            if trace is not None and trace.vec is not None
        ]
        if not traces:
            raise RuntimeError("Global PCA search requires at least one stored trace.")

        mat = np.stack([np.asarray(trace.vec, dtype="float32") for trace in traces], axis=0)
        self.trace_ids = [str(trace.trace_id) for trace in traces]
        self.mean = mat.mean(axis=0).astype("float32")
        centered = mat - self.mean

        _, _, vt = np.linalg.svd(centered.astype("float64"), full_matrices=False)
        basis_rows = min(int(pca_dim), vt.shape[0])
        self.basis = vt[:basis_rows].astype("float32")

        projected = centered @ self.basis.T
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
        projected = (projected / norms).astype("float32")

        self.index = faiss.IndexFlatIP(projected.shape[1])
        self.index.add(projected)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        projected_q = (np.asarray(query_vec, dtype="float32") - self.mean) @ self.basis.T
        projected_q /= (np.linalg.norm(projected_q) + 1e-8)
        k = min(top_k, len(self.trace_ids))
        scores, indices = self.index.search(projected_q.reshape(1, -1).astype("float32"), k)

        hits: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.trace_ids):
                hits.append((self.trace_ids[idx], float(score)))
        return hits


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
    router: Optional[GlyphRouter] = None
    project: str = "default"
    _global_pca_cache: Optional[Tuple[int, _GlobalPCAIndex]] = None

    # --- Shared encoder/instance setup ---
    @staticmethod
    def _resolve_encoders(encoder_mode, vec_dim, use_clip=True,
                          model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        """Resolve text/image encoders and output dimension from mode string."""
        text_enc = None
        img_enc = None
        store_dim = vec_dim

        if encoder_mode == "minilm":
            text_enc = TextMiniLM()
            store_dim = 384
        elif encoder_mode == "hash":
            text_enc = TextHasher(dim=vec_dim)
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
            text_enc = TextHasher(dim=vec_dim)

        if img_enc is None:
            img_enc = ImageStub(dim=store_dim)

        return text_enc, img_enc, store_dim

    @classmethod
    def _build_instance(
        cls,
        store,
        text_enc,
        img_enc,
        use_gravity=True,
        field=None,
        router_use_projection: bool = False,
    ):
        """Build a Hologram instance from components (shared by init/load)."""
        if field is None and use_gravity:
            field = GravityField(dim=store.vec_dim)
        manifold = LatentManifold(dim=store.vec_dim)
        glyphs_reg = GlyphRegistry(store)
        router = GlyphRouter(
            store,
            glyphs_reg,
            gravity_field=field,
            use_projection=router_use_projection,
        )
        return cls(
            store=store, glyphs=glyphs_reg,
            text_encoder=text_enc, image_encoder=img_enc,
            manifold=manifold, field=field, router=router,
        )

    @staticmethod
    def _auto_ingest_system(instance, auto_ingest, use_gravity, text_enc):
        """Ingest system concepts as Tier 2 if not already present."""
        if not (auto_ingest and use_gravity and instance.field):
            return
        from .system_kb import get_system_concepts
        from .text_utils import extract_concepts

        existing = [n for n in instance.field.sim.concepts
                    if n.startswith("system:")
                    and instance.field.sim.concepts[n].tier == 2]
        if existing:
            return

        for concept_text in extract_concepts(get_system_concepts()):
            vec = instance.manifold.align_text(concept_text, text_enc)
            cid = _stable_id("system", concept_text)
            instance.field.add(cid, vec, tier=2, project="hologram",
                               origin="system_design")

    def _invalidate_search_caches(self) -> None:
        self._global_pca_cache = None
        if self.router is not None:
            self.router.invalidate()

    def _get_global_pca_index(self, pca_dim: int = 64) -> _GlobalPCAIndex:
        cached = self._global_pca_cache
        if cached is not None and cached[0] == pca_dim:
            return cached[1]
        index = _GlobalPCAIndex(self.store, pca_dim=pca_dim)
        self._global_pca_cache = (pca_dim, index)
        return index

    # --- Initialization ---
    @classmethod
    def init(
        cls,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        use_clip: bool = True,
        use_gravity: bool = True,
        encoder_mode: str = "minilm",
        auto_ingest_system: bool = True,
        router_use_projection: bool = False,
    ):
        text_enc, img_enc, store_dim = cls._resolve_encoders(
            encoder_mode, VECTOR_DIM, use_clip, model_name, pretrained)
        store = MemoryStore(vec_dim=store_dim)
        instance = cls._build_instance(
            store,
            text_enc,
            img_enc,
            use_gravity,
            router_use_projection=router_use_projection,
        )
        
        # NEW: Initialize Normalization Pipeline
        from .normalization import NormalizationPipeline
        from .text_utils import set_global_pipeline
        
        # Define encode function for manifold alignment
        def encode_func(text: str) -> np.ndarray:
            return instance.manifold.align_text(text, instance.text_encoder)
            
        pipeline = NormalizationPipeline(
            gravity_field=instance.field,
            encode_func=encode_func,
            enable_llm_correction=False
        )
        set_global_pipeline(pipeline)

        # Auto-ingest system concepts as Tier 2
        cls._auto_ingest_system(instance, auto_ingest_system, use_gravity, text_enc)

        return instance

    # --- Write operations ---
    def add_text(self, glyph_id: str, text: str, trace_id: Optional[str] = None, do_extract_concepts: bool = False, add_to_field: bool = True, 
                 # NEW parameters
                 tier: int = 1,  # Default to Tier 1 (Domain)
                 origin: str = "kb",  # "kb", "runtime", "manual", "system_design"
                 **meta):
        
        # 1. Normalize text (cleaning + spelling + fuzzy resolution)
        skip_nlp = meta.get("skip_nlp", False)
        
        raw_text = text
        canonical_trace_id = None
        
        if not skip_nlp:
            text, canonical_trace_id = normalize_text(text, store=self.store, encoder=self.text_encoder)
            
            if text != raw_text:
                meta["raw_text"] = raw_text

        # --- Coreference Resolution ---
        from .config import Config
        from .coref import resolve
        
        resolved_text = text
        coref_map = {}
        
        if not skip_nlp and Config.coref.ENABLE_COREF:
            # 1. Structural Resolution
            # Pass NORMALIZED text to coref
            resolved_text, coref_map = resolve(text)
            
            # 2. Gravity Fallback
            normalized_pronouns = ["this", "that", "it", "these", "those"]
            
            if Config.coref.ENABLE_GRAVITY_FALLBACK and self.field:
                for word in text.split():
                    w_clean = word.lower().strip(".,!?")
                    if w_clean in normalized_pronouns and word not in coref_map:
                        # Try gravity resolution
                        antecedent = self.field.resolve_pronoun(text, word)
                        if antecedent:
                            coref_map[word] = antecedent

        trace_id = trace_id or _stable_id("text", text)
        # Use manifold for alignment
        vec = self.manifold.align_text(text, self.text_encoder)
        
        tr = Trace(
            trace_id=trace_id, 
            kind="text", 
            content=text, # Store normalized text
            vec=vec, 
            meta=meta,
            resolved_text=resolved_text,
            coref_map=coref_map
        )
        self.glyphs.attach_trace(glyph_id, tr)
        if self.field and add_to_field:
            self.field.add(
                trace_id, 
                vec,
                tier=tier,
                project=self.project,
                origin=origin
            )
            
            # If fuzzy resolution found a canonical, trigger fusion
            if canonical_trace_id and canonical_trace_id != trace_id:
                self.field.sim.fuse_concepts(trace_id, canonical_trace_id, transfer_mass=True)
            
            if do_extract_concepts:
                concepts = extract_concepts(resolved_text)
                for concept_text in concepts:
                    # Normalize concept text too (fuzzy resolve concepts to existing nodes)
                    concept_text_normalized, concept_canonical_id = normalize_text(
                        concept_text, store=self.store, encoder=self.text_encoder
                    )
                    
                    # Concepts also go through manifold
                    c_vec = self.manifold.align_text(concept_text_normalized, self.text_encoder)
                    c_id = _stable_id("concept", concept_text_normalized)
                    self.field.add(
                        c_id, 
                        vec=c_vec,
                        tier=tier,
                        project=self.project,
                        origin=origin
                    )
                    
                    # Trigger fusion if needed
                    if concept_canonical_id and concept_canonical_id != c_id:
                        self.field.sim.fuse_concepts(c_id, concept_canonical_id, transfer_mass=True)
            
            # Trigger dynamic self-regulation (Auto-Fusion & Auto-Mitosis)
            self.field.sim.step_dynamics()

        # Invalidate glyph router shards (trace membership changed)
        self._invalidate_search_caches()

        return trace_id

    def add_image_path(self, glyph_id: str, path: str, trace_id: Optional[str] = None, **meta):
        trace_id = trace_id or _stable_id("image", path)
        # Use manifold for alignment
        vec = self.manifold.align_image(path, self.image_encoder)
        
        tr = Trace(trace_id=trace_id, kind="image", content=path, vec=vec, meta=meta)
        self.glyphs.attach_trace(glyph_id, tr)
        if self.field:
            self.field.add(trace_id, vec)
        self._invalidate_search_caches()
        return trace_id

    def ingest_document(self, glyph_id: str, text: str,
                        sentences_per_chunk: int = 3, overlap: int = 1,
                        tier: int = 1, origin: str = "kb",
                        normalize: bool = True,
                        base_meta: Optional[Dict[str, object]] = None,
                        ) -> List[dict]:
        """Chunk text, batch embed, and store as traces in one pass.

        Higher-level alternative to add_text() for bulk document ingestion.
        Applies normalization + coreference by default (same quality as add_text).
        Returns list of dicts with trace_id and chunk metadata.

        Args:
            normalize: If True, run normalization + coref on each chunk before
                       embedding. Set False for pre-cleaned text.
        """
        chunks = chunk_text(text, sentences_per_chunk=sentences_per_chunk,
                            overlap=overlap)
        if not chunks:
            return []

        # Apply normalization + coref per chunk if requested
        chunk_texts = []
        for c in chunks:
            ct = c.text
            if normalize:
                ct, _ = normalize_text(ct, store=self.store,
                                       encoder=self.text_encoder)
                from .config import Config
                from .coref import resolve
                if Config.coref.ENABLE_COREF:
                    ct, _ = resolve(ct)
            chunk_texts.append(ct)

        # Batch embed all (possibly normalized) chunk texts
        has_batch = hasattr(self.text_encoder, "encode_batch")
        if has_batch:
            vecs = self.text_encoder.encode_batch(chunk_texts)
        else:
            vecs = np.stack([
                self.manifold.align_text(t, self.text_encoder) for t in chunk_texts
            ])

        # Content-idempotent: skip chunks that already exist in the store.
        # Same document → same source_hash → same chunk IDs → no-op.
        # Changed document → new source_hash → new IDs → ingests as new.
        results = []
        any_new = False
        inherited_meta = dict(base_meta or {})
        for chunk, ct, vec in zip(chunks, chunk_texts, vecs):
            tid = f"chunk:{chunk.source_hash}:{chunk.index}"
            if self.store.get_trace(tid) is not None:
                results.append({"trace_id": tid, "chunk_index": chunk.index,
                                "char_start": chunk.char_start,
                                "char_end": chunk.char_end})
                continue

            vec = vec / (np.linalg.norm(vec) + 1e-8)
            tr = Trace(
                trace_id=tid, kind="chunk", content=ct, vec=vec,
                meta={
                    **inherited_meta,
                    "chunk_index": chunk.index,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                    "sentence_start": chunk.sentence_start,
                    "sentence_end": chunk.sentence_end,
                    "source_hash": chunk.source_hash,
                },
            )
            self.glyphs.attach_trace(glyph_id, tr)
            if self.field:
                self.field.add(tid, vec, tier=tier, project=self.project,
                               origin=origin)
            any_new = True
            results.append({"trace_id": tid, "chunk_index": chunk.index,
                            "char_start": chunk.char_start,
                            "char_end": chunk.char_end})

        # Invalidate router only if new chunks were added
        if any_new:
            self._invalidate_search_caches()

        return results

    def ingest_parsed_document(
        self,
        glyph_id: str,
        parsed: ParsedDocument,
        sentences_per_chunk: int = 3,
        overlap: int = 1,
        tier: int = 1,
        origin: str = "kb",
        normalize: bool = True,
        ingest_images: bool = True,
    ) -> Dict[str, object]:
        """Ingest a parsed document into text chunks and optional image traces."""
        if self.store.get_glyph(glyph_id) is None:
            self.glyphs.create(glyph_id, title=glyph_id)

        all_chunks: List[dict] = []
        image_traces: List[dict] = []
        for page in parsed.pages:
            page_meta = {
                "source_doc": parsed.doc_id,
                "source_path": parsed.source_path,
                "page_number": page.page_number,
                **parsed.metadata,
                **page.metadata,
            }
            if page.text.strip():
                chunks = self.ingest_document(
                    glyph_id=glyph_id,
                    text=page.text,
                    sentences_per_chunk=sentences_per_chunk,
                    overlap=overlap,
                    tier=tier,
                    origin=origin,
                    normalize=normalize,
                    base_meta=page_meta,
                )
                all_chunks.extend(chunks)

            if ingest_images:
                for image in page.images:
                    image_trace_id = self.add_image_path(
                        glyph_id,
                        image.path,
                        trace_id=f"image:{parsed.doc_id}:{page.page_number}:{image.image_index}",
                        source_doc=parsed.doc_id,
                        source_path=parsed.source_path,
                        page_number=page.page_number,
                        figure_id=f"{parsed.doc_id}:p{page.page_number}:img{image.image_index}",
                        bbox=image.bbox,
                        caption_text=image.caption_text,
                        **image.metadata,
                    )
                    image_traces.append(
                        {
                            "trace_id": image_trace_id,
                            "path": image.path,
                            "page_number": page.page_number,
                            "image_index": image.image_index,
                        }
                    )

        return {
            "doc_id": parsed.doc_id,
            "pages_ingested": len(parsed.pages),
            "chunks": all_chunks,
            "images": image_traces,
        }

    def ingest_file(
        self,
        glyph_id: str,
        path: str,
        sentences_per_chunk: int = 3,
        overlap: int = 1,
        tier: int = 1,
        origin: str = "kb",
        normalize: bool = True,
        ingest_images: bool = True,
        image_output_dir: Optional[str] = None,
    ) -> Dict[str, object]:
        """Parse a supported document file and ingest it into the current glyph."""
        file_path = Path(path).expanduser().resolve()
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            parsed = parse_pdf(
                str(file_path),
                image_output_dir=image_output_dir,
                extract_images=ingest_images,
            )
        elif suffix == ".docx":
            from hologram.parsers import parse_docx
            parsed = parse_docx(
                str(file_path),
                image_output_dir=image_output_dir,
                extract_images=ingest_images,
            )
        else:
            raise ValueError(
                f"Unsupported file type for ingest_file: {suffix or '<no extension>'}"
            )

        return self.ingest_parsed_document(
            glyph_id=glyph_id,
            parsed=parsed,
            sentences_per_chunk=sentences_per_chunk,
            overlap=overlap,
            tier=tier,
            origin=origin,
            normalize=normalize,
            ingest_images=ingest_images,
        )

    def ingest_code(self, file_path: str) -> int:
        """
        Ingest source code file (read from disk).
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.ingest_code_content(file_path, content)

    def ingest_code_content(self, file_path: str, content: str, add_to_field: bool = True):
        """
        Ingest source code directly from string content.
        """
        from .code_map.evolution import CodeEvolutionEngine
        
        def vectorizer(text: str) -> np.ndarray:
             return self.manifold.align_text(text, self.text_encoder)
             
        engine = CodeEvolutionEngine(self.store, vectorizer)
        
        # We need to bypass file reading.
        # Check if engine has process_source or similar.
        # If not, we might need to rely on engine implementation details.
        # Assuming we can add/use a method that takes content.
        # Let's assume process_source exists or we add it.
        if hasattr(engine, 'process_source'):
            count = engine.process_source(content, file_path)
        else:
            # Fallback: Create temp file? Or raise error?
            # Ideally, update CodeEvolutionEngine.
            # For now, let's assume we update CodeEvolutionEngine to have process_source.
            raise NotImplementedError("CodeEvolutionEngine needs process_source method")
            
        if self.field:
            self.field.sim.step_dynamics()
            
        return count

    # --- Read operations ---
    def recall_glyph(self, glyph_id: str):
        return self.glyphs.recall(glyph_id)

    def search_text(self, query: str, top_k: int = 5):
        # Use manifold for alignment
        qv = self.manifold.align_text(query, self.text_encoder)
        return self.glyphs.search_across(qv, top_k=top_k)

    def search_global_pca(self, query: str, top_k: int = 5, pca_dim: int = 64) -> List[Tuple[Trace, float]]:
        qv = self.manifold.align_text(query, self.text_encoder)
        hits = self._get_global_pca_index(pca_dim=pca_dim).search(qv, top_k=top_k)
        out = []
        for trace_id, score in hits:
            t = self.store.get_trace(trace_id)
            if t:
                out.append((t, score))
        return out

    def search_routed(
        self,
        query: str,
        top_k: int = 5,
        top_glyphs: int = 2,
        secondary_shard_weight: float = 1.0,
        shard2_cutoff_rank: Optional[int] = None,
    ) -> List[Tuple[Trace, float]]:
        """Glyph-routed retrieval: infer glyphs → search shards → merge.

        Returns same format as search_text() for easy A/B comparison.
        """
        if self.router is None:
            return self.glyphs.search_across(
                self.manifold.align_text(query, self.text_encoder), top_k=top_k)

        qv = self.manifold.align_text(query, self.text_encoder)
        hits = self.router.search_routed(
            qv,
            top_k=top_k,
            top_glyphs=top_glyphs,
            secondary_shard_weight=secondary_shard_weight,
            shard2_cutoff_rank=shard2_cutoff_rank,
        )
        # Convert (trace_id, score) → (Trace, score) to match search_text format
        out = []
        for trace_id, score in hits:
            t = self.store.get_trace(trace_id)
            if t:
                out.append((t, score))
        return out

    def search_adaptive(self, query: str, top_k: int = 5,
                        top_glyphs: int = 2,
                        secondary_shard_weight: float = 1.0,
                        shard2_cutoff_rank: Optional[int] = None) -> List[Tuple[Trace, float]]:
        """Adaptive retrieval: stay global unless shard population warrants routing."""
        qv = self.manifold.align_text(query, self.text_encoder)
        if self.router is None:
            return self.glyphs.search_across(qv, top_k=top_k)

        hits = self.router.search_adaptive(
            qv,
            top_k=top_k,
            top_glyphs=top_glyphs,
            secondary_shard_weight=secondary_shard_weight,
            shard2_cutoff_rank=shard2_cutoff_rank,
        )
        out = []
        for trace_id, score in hits:
            t = self.store.get_trace(trace_id)
            if t:
                out.append((t, score))
        return out

    def choose_dynamic_strategy(
        self,
        optimize_for: str = "balanced",
        global_pca_max_traces: int = 5000,
        min_routable_glyphs: int = 2,
        min_traces_per_glyph: int = 3,
        min_total_traces: int = 8,
    ) -> str:
        """Choose a retrieval path based on current scale and optimization target.

        Policy:
        - `quality`: use full global search
        - `speed`/`balanced`: use global PCA while corpus is small enough;
          switch to routed retrieval once scale exceeds the PCA threshold and
          glyph routing is actually viable.
        """
        if optimize_for not in {"quality", "balanced", "speed"}:
            raise ValueError("optimize_for must be one of: quality, balanced, speed")

        if optimize_for == "quality":
            return "global"

        total_traces = len(self.store.traces)
        if total_traces <= global_pca_max_traces:
            return "global_pca"

        if self.router is None:
            return "global_pca"

        decision = self.router.decide_routing(
            min_routable_glyphs=min_routable_glyphs,
            min_traces_per_glyph=min_traces_per_glyph,
            min_total_traces=min_total_traces,
        )
        return "routed" if decision.should_route else "global_pca"

    def search_dynamic(
        self,
        query: str,
        top_k: int = 5,
        optimize_for: str = "balanced",
        top_glyphs: int = 2,
        global_pca_dim: int = 64,
        global_pca_max_traces: int = 5000,
        secondary_shard_weight: float = 0.9,
        shard2_cutoff_rank: Optional[int] = 7,
    ) -> Tuple[str, List[Tuple[Trace, float]]]:
        """Dynamically choose between quality-first and speed-first retrieval paths."""
        strategy = self.choose_dynamic_strategy(
            optimize_for=optimize_for,
            global_pca_max_traces=global_pca_max_traces,
        )
        if strategy == "global":
            return strategy, self.search_text(query, top_k=top_k)
        if strategy == "global_pca":
            return strategy, self.search_global_pca(query, top_k=top_k, pca_dim=global_pca_dim)
        return strategy, self.search_routed(
            query,
            top_k=top_k,
            top_glyphs=top_glyphs,
            secondary_shard_weight=secondary_shard_weight,
            shard2_cutoff_rank=shard2_cutoff_rank,
        )

    def search_image_path(self, path: str, top_k: int = 5):
        # Use manifold for alignment
        qv = self.manifold.align_image(path, self.image_encoder)
        return self.glyphs.search_across(qv, top_k=top_k)

    def query_code(self, query: str, top_k: int = 5):
        """
        Retrieve code concepts matching the query.
        """
        qv = self.manifold.align_text(query, self.text_encoder)
        
        # Search traces
        hits = self.store.search_traces(qv, top_k=top_k * 5) # Fetch more, filter later
        
        results = []
        for trace_id, score in hits:
            trace = self.store.get_trace(trace_id)
            if trace and trace.kind == "code":
                results.append({
                    "concept": trace.meta.get("code_type", "symbol") + ":" + trace_id.split(":")[-1],
                    "file": trace.source_file,
                    "span": trace.span,
                    "score": float(score),
                    "snippet": trace.content[:200]
                })
                
        return sorted(results, key=lambda x: x["score"], reverse=True)[:top_k]


    def search_with_drift(
        self,
        query: str,
        top_k_traces: int = 10,
        probe_steps: int = 8,
    ) -> Dict[str, Any]:
        """
        Perform a dynamic retrieval using probe physics (Phase 4).
        Returns:
            {
                "probe": None,  # Legacy field
                "tree": RetrievalTree,
                "results": List[Dict]  # Ranked traces
            }
        """
        # 1. Encode query via Manifold
        qv = self.manifold.align_text(query, self.text_encoder)

        if not self.field:
            # Fallback to standard search if no gravity field
            hits = self.store.search_traces(qv, top_k=top_k_traces)
            results = []
            for tid, s in hits:
                t = self.store.get_trace(tid)
                if t: results.append({"trace": t, "score": float(s)})
            return {"probe": None, "tree": None, "results": results}

        # 2. Run ProbeRetriever
        from .probe import ProbeRetriever
        from .cost_engine import CostEngine
        
        # Initialize CostEngine for this retrieval 
        # (Could be cached on self, but cheap to init structure)
        cost_engine = CostEngine()
        
        retriever = ProbeRetriever(self.field, cost_engine)
        tree = retriever.retrieve_tree(
            query_vec=qv,
            top_k_seeds=100,
            max_depth=3,
            final_k=40
        )

        # 3. Map concepts -> traces
        # Traces are either direct text nodes (concept_id="trace:...") or linked via Glyphs
        trace_scores: Dict[str, float] = {}
        
        # The tree nodes are our anchor points.
        # We need to rank them by relevance to the query.
        # This information is in node.sim_to_query and node.path_energy.
        
        for nid, node in tree.nodes.items():
            
            # Case A: Concept IS a trace
            tr = self.store.get_trace(nid)
            if tr:
                trace_scores[nid] = max(trace_scores.get(nid, 0.0), node.sim_to_query)
            
            # Case B: Concept is a Glyph -> get attached traces
            if nid.startswith("glyph:"):
                # Glyph ID is the part after "glyph:"
                gid = nid.replace("glyph:", "")
                glyph = self.store.get_glyph(gid)
                if glyph:
                    for tid in glyph.trace_ids:
                        trace_scores[tid] = max(trace_scores.get(tid, 0.0), node.sim_to_query)

        # 4. Rank traces
        ranked = sorted(
            trace_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_traces]

        results = []
        for tid, s in ranked:
            tr = self.store.get_trace(tid)
            if tr:
                results.append({
                    "trace": tr,
                    "score": s,
                })

        # 5. Return both path + tree + traces
        return {
            "probe": None, # Deprecated legacy field
            "tree": tree,
            "results": results,
        }

    def retrieve(self, query: str) -> MemoryPacket:
        """
        Perform a dynamic retrieval using probe physics.
        Returns a structured Memory Packet (SMI).
        """
        # Use the new Phase 4 Dynamic Graph Retrieval
        drift_result = self.search_with_drift(query)
        
        tree = drift_result["tree"]
        
        nodes = []
        glyphs = []
        edges = [] 
        
        if tree:
            # Flatten tree nodes into list
            for nid, node in tree.nodes.items():
                # Node Dict for MemoryPacket
                n_dict = {
                    "name": nid,
                    "mass": round(node.mass, 3),
                    "score": round(node.sim_to_query, 3), # Use similarity as score
                    "energy": round(node.path_energy, 3) 
                    # "age": ... # Optional if available
                }
                nodes.append(n_dict)
                
                if nid.startswith("glyph:"):
                    glyphs.append({
                        "id": nid.replace("glyph:", ""),
                        "mass": round(node.mass, 3),
                        "similarity": round(node.sim_to_query, 3)
                    })
            
            # Edges from tree structure
            for edge in tree.edges:
                 edges.append({
                     "a": edge.source,
                     "b": edge.target,
                     "relation": round(edge.relation_strength, 3),
                     "tension": round(edge.edge_energy, 3) # Use edge energy as tension proxy
                 })
        
        # Wrap in Memory Packet
        packet = MemoryPacket(
            seed=query,
            nodes=nodes,
            edges=edges,
            glyphs=glyphs,
            trajectory_steps=0 # Probe logic changed, no linear trajectory
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

    # --- KG + Drift ---
    def build_kg_batch(
        self,
        batch_id: str,
        items: List[Dict[str, Any]],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a semantic batch knowledge graph snapshot.
        """
        from .kg.builder import build_batch_kg_snapshot

        snapshot = build_batch_kg_snapshot(batch_id=batch_id, items=items, timestamp=timestamp)
        return snapshot.to_dict()

    def compare_drift(
        self,
        baseline_id: str,
        target_id: str,
        baseline_items: List[Dict[str, Any]],
        target_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare two batches and return drift report.
        """
        from .drift.engine import compare_batches
        from .drift.models import DriftComparisonInput
        from .kg.builder import build_batch_kg_snapshot

        comparison = DriftComparisonInput(
            baseline_id=baseline_id,
            target_id=target_id,
            baseline_items=baseline_items,
            target_items=target_items,
        )

        baseline_snapshot = build_batch_kg_snapshot(baseline_id, baseline_items)
        target_snapshot = build_batch_kg_snapshot(target_id, target_items)

        def embed_func(text: str) -> np.ndarray:
            return self.manifold.align_text(text, self.text_encoder)

        report = compare_batches(
            comparison,
            embed_func=embed_func,
            baseline_snapshot=baseline_snapshot,
            target_snapshot=target_snapshot,
        )
        payload = report.to_dict()
        payload["baseline_snapshot"] = baseline_snapshot.to_dict()
        payload["target_snapshot"] = target_snapshot.to_dict()
        return payload

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
        encoder_mode: str = "minilm",
        auto_ingest_system: bool = True,  # NEW: Auto-load system concepts
        router_use_projection: bool = False,
    ):
        store = MemoryStore.load(path)
        text_enc, img_enc, _ = cls._resolve_encoders(
            encoder_mode, store.vec_dim, use_clip, model_name, pretrained)

        # Restore gravity state from store if available
        field = None
        if use_gravity and hasattr(store, 'sim') and store.sim is not None:
            field = GravityField(dim=store.vec_dim)
            field.sim = store.sim
        elif use_gravity:
            field = GravityField(dim=store.vec_dim)

        instance = cls._build_instance(
            store,
            text_enc,
            img_enc,
            use_gravity,
            field,
            router_use_projection=router_use_projection,
        )
        cls._auto_ingest_system(instance, auto_ingest_system, use_gravity, text_enc)
        return instance
