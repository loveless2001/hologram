#!/usr/bin/env python3
"""Benchmark mixed-domain retrieval on a union of LoTTE domains.

This is intended to stress retrieval on real long-tail data with explicit
cross-domain competition:
- each LoTTE domain is ingested as its own glyph/domain
- queries run against the full mixed corpus
- global retrieval competes across all domains
- routed/adaptive retrieval can exploit domain shards
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import ir_datasets
import numpy as np
import faiss

from hologram.api import Hologram
from hologram.glyph_router import GlyphRouter
from hologram.store import Trace

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, Dict[str, float]]

TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def compute_metrics(qrels: Qrels, results: Results, ks: List[int]) -> Dict[str, Dict[str, float]]:
    from beir.retrieval.evaluation import EvaluateRetrieval

    ndcg, map_at_k, recall, precision = EvaluateRetrieval.evaluate(
        qrels=qrels,
        results=results,
        k_values=ks,
    )
    mrr_raw = EvaluateRetrieval.evaluate_custom(
        qrels=qrels,
        results=results,
        k_values=ks,
        metric="mrr",
    )
    mrr = mrr_raw[0] if isinstance(mrr_raw, tuple) else mrr_raw
    return {
        "ndcg": ndcg,
        "recall": recall,
        "precision": precision,
        "map": map_at_k,
        "mrr": mrr,
    }


def _dcg(relevances: List[int], k: int) -> float:
    score = 0.0
    for idx, rel in enumerate(relevances[:k], start=1):
        if rel <= 0:
            continue
        score += (2 ** rel - 1) / np.log2(idx + 1)
    return float(score)


def _query_domain(qid: str) -> str:
    return qid.split(":q:", 1)[0]


def compute_per_query_metrics(
    qrels: Qrels,
    results: Results,
    ks: List[int],
) -> Dict[str, Dict[str, float | int | str]]:
    per_query: Dict[str, Dict[str, float | int | str]] = {}
    for qid, rels in qrels.items():
        ranked = sorted(results.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
        relevant_total = sum(1 for rel in rels.values() if rel > 0)
        row: Dict[str, float | int | str] = {
            "query_id": qid,
            "domain": _query_domain(qid),
            "relevant_total": relevant_total,
        }
        ranked_doc_ids = [doc_id for doc_id, _ in ranked]
        ranked_rels = [int(rels.get(doc_id, 0)) for doc_id in ranked_doc_ids]
        ideal_rels = sorted((int(v) for v in rels.values()), reverse=True)
        rr = 0.0
        ap_hits = 0
        ap_sum = 0.0
        for idx, rel in enumerate(ranked_rels[: max(ks)], start=1):
            if rel > 0:
                if rr == 0.0:
                    rr = 1.0 / idx
                ap_hits += 1
                ap_sum += ap_hits / idx
        for k in ks:
            hits_k = sum(1 for rel in ranked_rels[:k] if rel > 0)
            ideal_dcg = _dcg(ideal_rels, k)
            row[f"NDCG@{k}"] = _dcg(ranked_rels, k) / ideal_dcg if ideal_dcg > 0 else 0.0
            row[f"Recall@{k}"] = hits_k / relevant_total if relevant_total > 0 else 0.0
            row[f"P@{k}"] = hits_k / max(k, 1)
            row[f"MAP@{k}"] = ap_sum / relevant_total if relevant_total > 0 else 0.0
            row[f"MRR@{k}"] = rr
        per_query[qid] = row
    return per_query


def aggregate_per_domain(
    per_query: Dict[str, Dict[str, float | int | str]],
    ks: List[int],
) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[Dict[str, float | int | str]]] = {}
    for row in per_query.values():
        buckets.setdefault(str(row["domain"]), []).append(row)

    out: Dict[str, Dict[str, float]] = {}
    for domain, rows in buckets.items():
        agg: Dict[str, float] = {"queries": float(len(rows))}
        for metric in ("NDCG", "Recall", "P", "MAP", "MRR"):
            for k in ks:
                key = f"{metric}@{k}"
                agg[key] = float(sum(float(row[key]) for row in rows) / max(len(rows), 1))
        out[domain] = agg
    return out


def _iter_take(items: Iterable, limit: int | None):
    if limit is None:
        yield from items
        return
    for idx, item in enumerate(items):
        if idx >= limit:
            break
        yield item


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class BM25Index:
    """Minimal BM25 index for benchmark-time lexical retrieval."""

    def __init__(self, doc_rows: Sequence[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: Dict[str, int] = {}
        self.avgdl = 0.0
        self.doc_texts: Dict[str, str] = {}
        self.postings: Dict[str, List[Tuple[str, int]]] = {}
        self.doc_freqs: Dict[str, int] = {}

        total_len = 0
        for doc_id, text in doc_rows:
            self.doc_texts[doc_id] = text
            terms = _tokenize(text)
            self.doc_len[doc_id] = len(terms)
            total_len += len(terms)
            term_counts = Counter(terms)
            for term, tf in term_counts.items():
                self.postings.setdefault(term, []).append((doc_id, tf))
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.num_docs = len(doc_rows)
        self.avgdl = (total_len / self.num_docs) if self.num_docs else 0.0

    def search(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        if self.num_docs == 0:
            return []

        scores: Dict[str, float] = {}
        query_terms = _tokenize(query_text)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue
            df = self.doc_freqs[term]
            idf = math.log(1.0 + ((self.num_docs - df + 0.5) / (df + 0.5)))
            for doc_id, tf in postings:
                dl = self.doc_len[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-8)))
                score = idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-8))
                scores[doc_id] = scores.get(doc_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]


class PairReranker:
    """Optional cross-encoder reranker over (query, document) pairs."""

    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError("sentence-transformers is required for reranking.")
        self.model = CrossEncoder(model_name)

    def score(self, query_text: str, doc_texts: Sequence[str]) -> List[float]:
        if not doc_texts:
            return []
        pairs = [[query_text, doc_text] for doc_text in doc_texts]
        return [float(score) for score in self.model.predict(pairs)]


class GlobalPCARetriever:
    """Global PCA compression baseline over the mixed corpus."""

    def __init__(self, hg: Hologram, pca_dim: int):
        self.hg = hg
        self.pca_dim = int(pca_dim)
        self.mean: np.ndarray
        self.basis: np.ndarray
        self.index: faiss.IndexFlatIP
        self.trace_ids: List[str]
        self._glyph_indices: Dict[str, faiss.IndexFlatIP] = {}
        self._glyph_trace_ids: Dict[str, List[str]] = {}
        self._build()

    def _build(self) -> None:
        traces = [
            trace for trace in self.hg.store.traces.values()
            if trace is not None and trace.vec is not None
        ]
        if not traces:
            raise RuntimeError("GlobalPCARetriever requires at least one trace.")

        mat = np.stack([np.asarray(trace.vec, dtype="float32") for trace in traces], axis=0)
        self.trace_ids = [str(trace.trace_id) for trace in traces]
        self.mean = mat.mean(axis=0).astype("float32")
        centered = mat - self.mean

        _, _, vt = np.linalg.svd(centered.astype("float64"), full_matrices=False)
        basis_rows = min(self.pca_dim, vt.shape[0])
        self.basis = vt[:basis_rows].astype("float32")

        projected = centered @ self.basis.T
        norms = np.linalg.norm(projected, axis=1, keepdims=True) + 1e-8
        projected = (projected / norms).astype("float32")

        self.index = faiss.IndexFlatIP(projected.shape[1])
        self.index.add(projected)

        trace_to_row = {trace_id: idx for idx, trace_id in enumerate(self.trace_ids)}
        for glyph in self.hg.store.get_all_glyphs():
            glyph_trace_ids = [
                str(trace_id) for trace_id in glyph.trace_ids
                if str(trace_id) in trace_to_row
            ]
            if not glyph_trace_ids:
                continue
            rows = [trace_to_row[trace_id] for trace_id in glyph_trace_ids]
            glyph_mat = projected[rows]
            glyph_index = faiss.IndexFlatIP(glyph_mat.shape[1])
            glyph_index.add(glyph_mat)
            self._glyph_indices[glyph.glyph_id] = glyph_index
            self._glyph_trace_ids[glyph.glyph_id] = glyph_trace_ids

    def project_query_vec(self, query_vec: np.ndarray) -> np.ndarray:
        projected_q = (np.asarray(query_vec, dtype="float32") - self.mean) @ self.basis.T
        projected_q /= (np.linalg.norm(projected_q) + 1e-8)
        return projected_q.astype("float32")

    def search(self, query_text: str, top_k: int) -> List[Tuple[str, float]]:
        qv = self.hg.manifold.align_text(query_text, self.hg.text_encoder)
        projected_q = self.project_query_vec(qv)
        k = min(top_k, len(self.trace_ids))
        scores, indices = self.index.search(projected_q.reshape(1, -1).astype("float32"), k)

        hits: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.trace_ids):
                hits.append((self.trace_ids[idx], float(score)))
        return hits

    def search_glyphs(
        self,
        query_vec: np.ndarray,
        glyph_ids: Sequence[str],
        top_k: int,
    ) -> List[Tuple[str, float]]:
        projected_q = self.project_query_vec(query_vec)
        best_scores: Dict[str, float] = {}
        for glyph_id in glyph_ids:
            index = self._glyph_indices.get(glyph_id)
            trace_ids = self._glyph_trace_ids.get(glyph_id)
            if index is None or trace_ids is None:
                continue
            k = min(top_k, len(trace_ids))
            scores, indices = index.search(projected_q.reshape(1, -1), k)
            for score, idx in zip(scores[0], indices[0]):
                if 0 <= idx < len(trace_ids):
                    trace_id = trace_ids[idx]
                    if trace_id not in best_scores or float(score) > best_scores[trace_id]:
                        best_scores[trace_id] = float(score)
        return sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class WhiteningShardIndex:
    """Per-glyph full-rank whitening with shrinkage toward identity."""

    def __init__(self, hg: Hologram, shrinkage: float = 0.10, eig_floor: float = 1e-4):
        self.hg = hg
        self.shrinkage = float(min(max(shrinkage, 0.0), 1.0))
        self.eig_floor = max(float(eig_floor), 1e-8)
        self._indices: Dict[str, faiss.IndexFlatIP] = {}
        self._trace_ids: Dict[str, List[str]] = {}
        self._means: Dict[str, np.ndarray] = {}
        self._whiteners: Dict[str, np.ndarray] = {}
        self._build()

    def _build(self) -> None:
        for glyph in self.hg.store.get_all_glyphs():
            traces = [self.hg.store.get_trace(tid) for tid in glyph.trace_ids]
            traces = [trace for trace in traces if trace is not None and trace.vec is not None]
            if not traces:
                continue

            mat = np.stack([np.asarray(trace.vec, dtype="float32") for trace in traces], axis=0)
            mean = mat.mean(axis=0).astype("float32")
            centered = mat - mean
            dim = centered.shape[1]

            if len(traces) > 1:
                cov = (centered.T @ centered) / float(len(traces) - 1)
            else:
                cov = np.eye(dim, dtype="float32")

            avg_var = float(np.trace(cov) / max(dim, 1))
            identity_scale = avg_var if avg_var > 0.0 else 1.0
            shrunk = (1.0 - self.shrinkage) * cov + self.shrinkage * identity_scale * np.eye(dim, dtype="float32")

            eigvals, eigvecs = np.linalg.eigh(shrunk.astype("float64"))
            floor = max(identity_scale * self.eig_floor, self.eig_floor)
            eigvals = np.clip(eigvals, floor, None)
            whitener = (eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T).astype("float32")

            whitened = (centered @ whitener.T).astype("float32")
            norms = np.linalg.norm(whitened, axis=1, keepdims=True) + 1e-8
            whitened /= norms

            index = faiss.IndexFlatIP(dim)
            index.add(whitened)

            self._indices[glyph.glyph_id] = index
            self._trace_ids[glyph.glyph_id] = [str(trace.trace_id) for trace in traces]
            self._means[glyph.glyph_id] = mean
            self._whiteners[glyph.glyph_id] = whitener

    def search(self, glyph_id: str, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        index = self._indices.get(glyph_id)
        trace_ids = self._trace_ids.get(glyph_id)
        mean = self._means.get(glyph_id)
        whitener = self._whiteners.get(glyph_id)
        if index is None or trace_ids is None or mean is None or whitener is None:
            return []

        q = np.asarray(query_vec, dtype="float32") - mean
        q = (whitener @ q).astype("float32")
        q /= (np.linalg.norm(q) + 1e-8)
        k = min(top_k, len(trace_ids))
        scores, indices = index.search(q.reshape(1, -1), k)

        hits: List[Tuple[str, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(trace_ids):
                hits.append((trace_ids[idx], float(score)))
        return hits


def fuse_rrf(
    ranked_lists: Sequence[Sequence[Tuple[str, float]]],
    top_k: int,
    rrf_k: int,
) -> List[Tuple[str, float]]:
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def rerank_hits(
    reranker: Optional[PairReranker],
    query_text: str,
    hits: Sequence[Tuple[str, float]],
    doc_texts: Dict[str, str],
    rerank_top_n: int,
    top_k: int,
) -> List[Tuple[str, float]]:
    if reranker is None or not hits:
        return list(hits)[:top_k]

    limited = list(hits[: max(rerank_top_n, 1)])
    texts = [doc_texts[doc_id] for doc_id, _ in limited if doc_id in doc_texts]
    if len(texts) != len(limited):
        return list(hits)[:top_k]

    scores = reranker.score(query_text, texts)
    reranked = [(doc_id, score) for (doc_id, _), score in zip(limited, scores)]
    reranked.sort(key=lambda x: x[1], reverse=True)
    remainder = list(hits[max(rerank_top_n, 1):])
    return (reranked + remainder)[:top_k]


def ingest_domain(
    hg: Hologram,
    glyph_id: str,
    doc_rows: List[Tuple[str, str]],
) -> int:
    hg.glyphs.create(glyph_id, title=f"LoTTE {glyph_id.split(':', 1)[-1]}")

    texts = [text for _, text in doc_rows]
    if not texts:
        return 0

    if hasattr(hg.text_encoder, "encode_batch"):
        vecs = hg.text_encoder.encode_batch(texts)
        vecs = np.stack([hg.manifold.project(vec) for vec in vecs], axis=0)
    else:
        vecs = np.stack([hg.manifold.align_text(text, hg.text_encoder) for text in texts], axis=0)

    for (trace_id, text), vec in zip(doc_rows, vecs):
        trace = Trace(
            trace_id=trace_id,
            kind="text",
            content=text,
            vec=vec,
            meta={"skip_nlp": True, "origin": "benchmark"},
        )
        hg.store.add_trace(trace)
        hg.store.link_trace(glyph_id, trace_id)

    hg.glyphs._cache.pop(glyph_id, None)
    if hg.router is not None:
        hg.router.invalidate()

    return len(doc_rows)


def select_queries_and_qrels(
    domain: str,
    split: str,
    task: str,
    max_queries: int | None,
) -> Tuple[List[Tuple[str, str]], Qrels, set[str]]:
    dataset_name = f"lotte/{domain}/{split}/{task}"
    ds = ir_datasets.load(dataset_name)

    raw_qrels: Dict[str, Dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        prefixed_qid = f"{domain}:q:{qrel.query_id}"
        prefixed_doc_id = f"{domain}:doc:{qrel.doc_id}"
        raw_qrels.setdefault(prefixed_qid, {})[prefixed_doc_id] = int(qrel.relevance)

    kept_queries: List[Tuple[str, str]] = []
    filtered_qrels: Qrels = {}
    required_trace_ids: set[str] = set()
    for query in ds.queries_iter():
        prefixed_qid = f"{domain}:q:{query.query_id}"
        if prefixed_qid not in raw_qrels:
            continue
        kept_queries.append((prefixed_qid, query.text))
        filtered_qrels[prefixed_qid] = raw_qrels[prefixed_qid]
        required_trace_ids.update(raw_qrels[prefixed_qid].keys())
        if max_queries is not None and len(kept_queries) >= max_queries:
            break

    return kept_queries, filtered_qrels, required_trace_ids


def collect_doc_rows(
    domain: str,
    split: str,
    task: str,
    max_docs: int | None,
    required_trace_ids: set[str],
) -> Tuple[List[Tuple[str, str]], set[str]]:
    dataset_name = f"lotte/{domain}/{split}/{task}"
    ds = ir_datasets.load(dataset_name)

    required_doc_ids = {
        trace_id.split(":doc:", 1)[1]
        for trace_id in required_trace_ids
        if ":doc:" in trace_id
    }

    doc_rows: List[Tuple[str, str]] = []
    indexed_doc_ids: set[str] = set()
    filler_budget = None if max_docs is None else max(0, max_docs - len(required_doc_ids))
    filler_used = 0

    for doc in ds.docs_iter():
        doc_id = str(doc.doc_id)
        should_include = doc_id in required_doc_ids
        if not should_include and filler_budget is not None and filler_used < filler_budget:
            should_include = True
            filler_used += 1
        if not should_include:
            continue

        prefixed_doc_id = f"{domain}:doc:{doc_id}"
        doc_rows.append((prefixed_doc_id, doc.text.strip()))
        indexed_doc_ids.add(doc_id)

        if max_docs is not None and len(doc_rows) >= len(required_doc_ids) + filler_budget:
            break

    return doc_rows, indexed_doc_ids


def ingest_selected_domain(
    hg: Hologram,
    domain: str,
    split: str,
    task: str,
    max_docs: int | None,
    max_queries: int | None,
) -> Tuple[int, List[Tuple[str, str]], Qrels]:
    kept_queries, qrels, required_trace_ids = select_queries_and_qrels(
        domain=domain,
        split=split,
        task=task,
        max_queries=max_queries,
    )
    doc_rows, indexed_doc_ids = collect_doc_rows(
        domain=domain,
        split=split,
        task=task,
        max_docs=max_docs,
        required_trace_ids=required_trace_ids,
    )
    filtered_qrels = {
        qid: {
            doc_id: rel
            for doc_id, rel in rels.items()
            if doc_id.split(":doc:", 1)[1] in indexed_doc_ids
        }
        for qid, rels in qrels.items()
    }
    filtered_qrels = {qid: rels for qid, rels in filtered_qrels.items() if rels}
    kept_queries = [(qid, text) for qid, text in kept_queries if qid in filtered_qrels]
    indexed_docs = ingest_domain(hg=hg, glyph_id=f"lotte:{domain}", doc_rows=doc_rows)
    return indexed_docs, kept_queries, filtered_qrels


def dense_hits_for_query(
    hg: Hologram,
    qid: str,
    query_text: str,
    top_k: int,
    mode: str,
    top_glyphs: int,
    margin_threshold: float,
    identity_router: Optional[GlyphRouter],
    whitening_index: Optional[WhiteningShardIndex],
    secondary_shard_weight: float,
    shard2_cutoff_rank: int,
    global_pca_retriever: Optional[GlobalPCARetriever],
) -> List[Tuple[str, float]]:
    if mode == "global_pca":
        if global_pca_retriever is None:
            raise RuntimeError("global_pca mode requires a PCA retriever.")
        return global_pca_retriever.search(query_text, top_k=top_k)
    if mode == "routed_pca":
        if hg.router is None:
            raise RuntimeError("routed_pca mode requires hg.router to be available.")
        if global_pca_retriever is None:
            raise RuntimeError("routed_pca mode requires a PCA retriever.")
        hg.router._ensure_shards()
        qv = hg.manifold.align_text(query_text, hg.text_encoder)
        glyph_ids = list(hg.router.infer_glyphs(qv, top_n=top_glyphs).keys())
        hits = global_pca_retriever.search_glyphs(qv, glyph_ids=glyph_ids, top_k=top_k)
        if len(hits) < top_k:
            existing_ids = {trace_id for trace_id, _ in hits}
            for trace_id, score in global_pca_retriever.search(query_text, top_k=top_k):
                if trace_id not in existing_ids:
                    hits.append((trace_id, score))
                    if len(hits) >= top_k:
                        break
        return hits[:top_k]
    if mode == "oracle_pca":
        if global_pca_retriever is None:
            raise RuntimeError("oracle_pca mode requires a PCA retriever.")
        qv = hg.manifold.align_text(query_text, hg.text_encoder)
        glyph_id = f"lotte:{qid.split(':q:', 1)[0]}"
        return global_pca_retriever.search_glyphs(qv, glyph_ids=[glyph_id], top_k=top_k)
    if mode == "routed":
        hits = hg.search_routed(query_text, top_k=top_k, top_glyphs=top_glyphs)
    elif mode == "routed_weighted":
        hits = hg.search_routed(
            query_text,
            top_k=top_k,
            top_glyphs=top_glyphs,
            secondary_shard_weight=secondary_shard_weight,
        )
    elif mode == "routed_filtered":
        hits = hg.search_routed(
            query_text,
            top_k=top_k,
            top_glyphs=top_glyphs,
            secondary_shard_weight=secondary_shard_weight,
            shard2_cutoff_rank=shard2_cutoff_rank,
        )
    elif mode == "whitened_routed":
        if hg.router is None:
            raise RuntimeError("whitened_routed mode requires hg.router to be available.")
        if whitening_index is None:
            raise RuntimeError("whitened_routed mode requires a whitening index.")
        hg.router._ensure_shards()
        qv = hg.manifold.align_text(query_text, hg.text_encoder)
        glyph_weights = hg.router.infer_glyphs(qv, top_n=top_glyphs)
        best_scores: Dict[str, float] = {}
        for glyph_id in glyph_weights:
            for trace_id, score in whitening_index.search(glyph_id, qv, top_k=top_k):
                if trace_id not in best_scores or score > best_scores[trace_id]:
                    best_scores[trace_id] = score
        ranked_hits = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)
        if len(ranked_hits) < top_k:
            existing_ids = {trace_id for trace_id, _ in ranked_hits}
            for trace_id, score in hg.store.search_traces(qv, top_k=top_k):
                if trace_id not in existing_ids:
                    ranked_hits.append((trace_id, float(score)))
                    if len(ranked_hits) >= top_k:
                        break
        return ranked_hits[:top_k]
    elif mode == "routed_margin":
        if hg.router is None:
            raise RuntimeError("routed_margin mode requires hg.router to be available.")
        qv = hg.manifold.align_text(query_text, hg.text_encoder)
        glyph_weights = hg.router.infer_glyphs(qv, top_n=max(top_glyphs, 2))
        ranked = list(glyph_weights.items())
        gated_top_glyphs = top_glyphs
        if len(ranked) >= 2:
            margin = float(ranked[0][1] - ranked[1][1])
            if margin > margin_threshold:
                gated_top_glyphs = 1
        hits = hg.search_routed(query_text, top_k=top_k, top_glyphs=gated_top_glyphs)
    elif mode == "oracle":
        if hg.router is None:
            raise RuntimeError("Oracle mode requires hg.router to be available.")
        domain = qid.split(":q:", 1)[0]
        glyph_id = f"lotte:{domain}"
        hg.router._ensure_shards()
        shard = hg.router._shards.get(glyph_id)
        op = hg.router._operators.get(glyph_id)
        if shard is None or op is None:
            hits = []
        else:
            qv = hg.manifold.align_text(query_text, hg.text_encoder)
            transformed_q = op.transform_query(qv)
            shard_hits = shard.search(transformed_q, top_k=top_k)
            hits = []
            for trace_id, score in shard_hits:
                trace = hg.store.get_trace(trace_id)
                if trace is not None:
                    hits.append((trace, score))
    elif mode == "oracle_identity":
        if identity_router is None:
            raise RuntimeError("oracle_identity mode requires an identity router.")
        domain = qid.split(":q:", 1)[0]
        glyph_id = f"lotte:{domain}"
        shard = identity_router._shards.get(glyph_id)
        op = identity_router._operators.get(glyph_id)
        if shard is None or op is None:
            hits = []
        else:
            qv = hg.manifold.align_text(query_text, hg.text_encoder)
            transformed_q = op.transform_query(qv)
            shard_hits = shard.search(transformed_q, top_k=top_k)
            hits = []
            for trace_id, score in shard_hits:
                trace = hg.store.get_trace(trace_id)
                if trace is not None:
                    hits.append((trace, score))
    elif mode == "adaptive":
        hits = hg.search_adaptive(query_text, top_k=top_k, top_glyphs=top_glyphs)
    else:
        hits = hg.search_text(query_text, top_k=top_k)
    return [(str(trace.trace_id), float(score)) for trace, score in hits]


def run_queries(
    hg: Hologram,
    all_queries: List[Tuple[str, str]],
    top_k: int,
    mode: str,
    top_glyphs: int,
    margin_threshold: float,
    lexical_index: Optional[BM25Index] = None,
    lexical_top_k: int = 50,
    fusion: str = "none",
    rrf_k: int = 60,
    reranker: Optional[PairReranker] = None,
    rerank_top_n: int = 20,
    doc_texts: Optional[Dict[str, str]] = None,
    whitening_index: Optional[WhiteningShardIndex] = None,
    secondary_shard_weight: float = 1.0,
    shard2_cutoff_rank: int = 10,
    global_pca_retriever: Optional[GlobalPCARetriever] = None,
) -> Results:
    results: Results = {}
    identity_router = None
    if mode == "oracle_identity":
        identity_router = GlyphRouter(
            hg.store,
            hg.glyphs,
            gravity_field=hg.field,
            use_projection=False,
        )
        identity_router._ensure_shards()
    for qid, query_text in all_queries:
        dense_hits = dense_hits_for_query(
            hg=hg,
            qid=qid,
            query_text=query_text,
            top_k=max(top_k, lexical_top_k, rerank_top_n),
            mode=mode,
            top_glyphs=top_glyphs,
            margin_threshold=margin_threshold,
            identity_router=identity_router,
            whitening_index=whitening_index,
            secondary_shard_weight=secondary_shard_weight,
            shard2_cutoff_rank=shard2_cutoff_rank,
            global_pca_retriever=global_pca_retriever,
        )

        if fusion == "rrf":
            if lexical_index is None:
                raise RuntimeError("fusion=rrf requires a lexical index.")
            lexical_hits = lexical_index.search(query_text, top_k=max(top_k, lexical_top_k, rerank_top_n))
            fused_hits = fuse_rrf(
                ranked_lists=[dense_hits, lexical_hits],
                top_k=max(top_k, rerank_top_n),
                rrf_k=rrf_k,
            )
            hits = rerank_hits(
                reranker=reranker,
                query_text=query_text,
                hits=fused_hits,
                doc_texts=doc_texts or {},
                rerank_top_n=rerank_top_n,
                top_k=top_k,
            )
        else:
            hits = rerank_hits(
                reranker=reranker,
                query_text=query_text,
                hits=dense_hits,
                doc_texts=doc_texts or {},
                rerank_top_n=rerank_top_n,
                top_k=top_k,
            )
        results[qid] = {doc_id: float(score) for doc_id, score in hits}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark routed retrieval on a mixed-domain LoTTE corpus")
    parser.add_argument(
        "--domains",
        default="lifestyle,recreation,science,technology,writing",
        help="Comma-separated LoTTE domains to union into one corpus",
    )
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--task", default="forum", choices=["forum", "search"])
    parser.add_argument("--encoder-mode", default="minilm", choices=["minilm", "hash", "clip", "default"])
    parser.add_argument(
        "--router-use-projection",
        action="store_true",
        help="Use glyph operator projection in the Hologram router instead of same-space shards.",
    )
    parser.add_argument(
        "--mode",
        choices=["global", "global_pca", "routed", "routed_pca", "routed_weighted", "routed_filtered", "whitened_routed", "routed_margin", "adaptive", "oracle", "oracle_identity", "oracle_pca"],
        default="global",
    )
    parser.add_argument(
        "--fusion",
        choices=["none", "rrf"],
        default="none",
        help="Fuse dense retrieval with lexical BM25 candidates using reciprocal rank fusion.",
    )
    parser.add_argument("--lexical-top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--reranker", choices=["none", "cross_encoder"], default="none")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-glyphs", type=int, default=2)
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=0.05,
        help="For routed_margin: if top1-top2 glyph score margin exceeds this, use only shard1.",
    )
    parser.add_argument("--max-docs-per-domain", type=int, default=1000)
    parser.add_argument("--max-queries-per-domain", type=int, default=50)
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--whitening-shrinkage", type=float, default=0.10)
    parser.add_argument("--whitening-eig-floor", type=float, default=1e-4)
    parser.add_argument(
        "--secondary-shard-weight",
        type=float,
        default=0.80,
        help="For routed_weighted: multiply non-top1 shard scores by this weight before merge.",
    )
    parser.add_argument(
        "--shard2-cutoff-rank",
        type=int,
        default=10,
        help="For routed_filtered: keep shard2 docs only if they beat shard1's rank-N score.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    domains = [d.strip() for d in args.domains.split(",") if d.strip()]
    if len(domains) < 2:
        raise SystemExit("Use at least two LoTTE domains so routing has real cross-domain competition.")

    hg = Hologram.init(
        encoder_mode=args.encoder_mode,
        use_clip=(args.encoder_mode in {"clip", "default"}),
        use_gravity=False,
        auto_ingest_system=False,
        router_use_projection=args.router_use_projection,
    )

    all_queries: List[Tuple[str, str]] = []
    qrels: Qrels = {}
    domain_sizes: Dict[str, Dict[str, int]] = {}

    ingest_start = time.time()
    for domain in domains:
        indexed_docs, kept_queries, domain_qrels = ingest_selected_domain(
            hg=hg,
            domain=domain,
            split=args.split,
            task=args.task,
            max_docs=args.max_docs_per_domain,
            max_queries=args.max_queries_per_domain,
        )
        all_queries.extend(kept_queries)
        qrels.update(domain_qrels)
        domain_sizes[domain] = {
            "indexed_docs": indexed_docs,
            "queries_kept": len(kept_queries),
        }
    ingest_s = time.time() - ingest_start

    if not all_queries or not qrels:
        raise SystemExit(
            "No overlapping queries/qrels after filtering. Increase --max-docs-per-domain "
            "or --max-queries-per-domain."
        )

    doc_texts = {
        trace_id: trace.content
        for trace_id, trace in hg.store.traces.items()
        if trace is not None and trace.content
    }
    lexical_index = BM25Index(list(doc_texts.items())) if args.fusion == "rrf" else None
    reranker = None
    if args.reranker == "cross_encoder":
        reranker = PairReranker(args.reranker_model)
    whitening_index = None
    if args.mode == "whitened_routed":
        whitening_index = WhiteningShardIndex(
            hg=hg,
            shrinkage=args.whitening_shrinkage,
            eig_floor=args.whitening_eig_floor,
        )
    global_pca_retriever = None
    if args.mode in {"global_pca", "routed_pca", "oracle_pca"}:
        global_pca_retriever = GlobalPCARetriever(hg=hg, pca_dim=args.pca_dim)

    search_start = time.time()
    results = run_queries(
        hg=hg,
        all_queries=all_queries,
        top_k=args.top_k,
        mode=args.mode,
        top_glyphs=args.top_glyphs,
        margin_threshold=args.margin_threshold,
        lexical_index=lexical_index,
        lexical_top_k=args.lexical_top_k,
        fusion=args.fusion,
        rrf_k=args.rrf_k,
        reranker=reranker,
        rerank_top_n=args.rerank_top_n,
        doc_texts=doc_texts,
        whitening_index=whitening_index,
        secondary_shard_weight=args.secondary_shard_weight,
        shard2_cutoff_rank=args.shard2_cutoff_rank,
        global_pca_retriever=global_pca_retriever,
    )
    search_s = time.time() - search_start

    ks = [1, 3, 5, 10]
    metrics = compute_metrics(qrels=qrels, results=results, ks=ks)
    per_query = compute_per_query_metrics(qrels=qrels, results=results, ks=ks)
    per_domain = aggregate_per_domain(per_query=per_query, ks=ks)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "LoTTE-MultiDomain",
        "domains": domains,
        "split": args.split,
        "task": args.task,
        "mode": args.mode,
        "config": {
            "top_k": args.top_k,
            "top_glyphs": args.top_glyphs,
            "max_docs_per_domain": args.max_docs_per_domain,
            "max_queries_per_domain": args.max_queries_per_domain,
            "encoder_mode": args.encoder_mode,
            "router_use_projection": args.router_use_projection,
            "pca_dim": args.pca_dim,
            "margin_threshold": args.margin_threshold,
            "fusion": args.fusion,
            "lexical_top_k": args.lexical_top_k,
            "rrf_k": args.rrf_k,
            "reranker": args.reranker,
            "reranker_model": args.reranker_model if args.reranker != "none" else None,
            "rerank_top_n": args.rerank_top_n,
            "whitening_shrinkage": args.whitening_shrinkage,
            "whitening_eig_floor": args.whitening_eig_floor,
            "secondary_shard_weight": args.secondary_shard_weight,
            "shard2_cutoff_rank": args.shard2_cutoff_rank,
        },
        "domain_sizes": domain_sizes,
        "timing": {
            "ingest_seconds": ingest_s,
            "query_seconds": search_s,
            "query_ms_per_query": (search_s / max(len(all_queries), 1)) * 1000.0,
        },
        "metrics": metrics,
        "per_domain_metrics": per_domain,
        "per_query": [per_query[qid] for qid, _ in all_queries],
        "queries_evaluated": len(all_queries),
    }

    output = Path(args.output) if args.output else Path(
        f"perf/lotte_multidomain_{args.task}_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Benchmark complete")
    print(f"Output: {output}")
    print(f"Domains: {', '.join(domains)}")
    print(f"Task: {args.task}")
    print(f"Queries evaluated: {len(all_queries)}")
    print(f"NDCG@10: {payload['metrics']['ndcg'].get('NDCG@10', 0.0):.4f}")
    print(f"Recall@10: {payload['metrics']['recall'].get('Recall@10', 0.0):.4f}")
    print(f"MRR@10: {payload['metrics']['mrr'].get('MRR@10', 0.0):.4f}")
    print(f"Query ms/query: {payload['timing']['query_ms_per_query']:.2f}")


if __name__ == "__main__":
    main()
