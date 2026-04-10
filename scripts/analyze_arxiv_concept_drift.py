#!/usr/bin/env python3
"""Concept drift analysis over arXiv time slices.

Strategy mirrors the TimeQA drift approach:
- semantic shift: 1 - cosine between concept vectors across time slices
- neighbor shift: Jaccard distance over top keywords across slices

Data source:
- arXiv Atom API (title + abstract + published timestamp)
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from hologram.embeddings import TextMiniLM


ARXIV_API = "http://export.arxiv.org/api/query"
NS = {"atom": "http://www.w3.org/2005/Atom"}

STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "by", "is", "are", "was", "were", "be", "been", "being", "it", "that", "this",
    "as", "from", "about", "into", "over", "after", "before", "than", "then",
    "which", "who", "whom", "what", "when", "where", "why", "how", "we", "our",
    "their", "using", "use", "used", "new", "paper", "results", "show", "model",
}


@dataclass
class Paper:
    title: str
    abstract: str
    published: str


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return 1.0 - (len(a & b) / len(u))


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())


def top_keywords(texts: List[str], k: int = 20) -> List[str]:
    counts: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            if tok in STOPWORDS or len(tok) < 3:
                continue
            counts[tok] = counts.get(tok, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in ranked[:k]]


def fetch_arxiv_slice(
    category: str,
    year_start: int,
    year_end: int,
    max_results: int,
    query_all: str | None,
    pause_s: float,
) -> List[Paper]:
    start = 0
    out: List[Paper] = []
    batch_size = min(100, max_results)

    # arXiv date range supports YYYYMMDDHHMM format
    date_q = f"submittedDate:[{year_start}01010000 TO {year_end}12312359]"
    search_parts = [f"cat:{category}", date_q]
    if query_all:
        search_parts.append(f"all:{query_all}")
    search_query = " AND ".join(search_parts)

    while len(out) < max_results:
        params = {
            "search_query": search_query,
            "start": str(start),
            "max_results": str(min(batch_size, max_results - len(out))),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        url = ARXIV_API + "?" + urllib.parse.urlencode(params)

        with urllib.request.urlopen(url, timeout=40) as resp:
            xml_data = resp.read()
        root = ET.fromstring(xml_data)
        entries = root.findall("atom:entry", NS)
        if not entries:
            break

        for e in entries:
            title = (e.findtext("atom:title", default="", namespaces=NS) or "").strip()
            summary = (e.findtext("atom:summary", default="", namespaces=NS) or "").strip()
            published = (e.findtext("atom:published", default="", namespaces=NS) or "").strip()
            if title and summary:
                out.append(Paper(title=title, abstract=summary, published=published))

        start += len(entries)
        if len(entries) < batch_size:
            break
        time.sleep(pause_s)

    return out[:max_results]


def cache_key(
    category: str,
    year_start: int,
    year_end: int,
    max_results: int,
    query_all: str | None,
) -> str:
    raw = f"{category}|{year_start}|{year_end}|{max_results}|{query_all or ''}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:14]


def encode_texts(
    texts: List[str],
    encoder: TextMiniLM,
    batch_size: int = 64,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 384), dtype="float32")
    try:
        vecs = encoder.model.encode(  # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")
    except Exception:
        return np.stack([encoder.encode(t) for t in texts]).astype("float32")


def concept_profile_for_slice(
    concept_vec: np.ndarray,
    slice_texts: List[str],
    slice_vecs: np.ndarray,
    top_m_docs: int,
    keyword_k: int,
) -> Tuple[np.ndarray, List[str], List[str]]:
    if slice_vecs.size == 0:
        return concept_vec, [], []

    sims = slice_vecs @ concept_vec.astype("float32")
    top_n = min(top_m_docs, len(slice_texts))
    idx = np.argsort(-sims)[:top_n]
    top_vecs = slice_vecs[idx]
    mean = top_vecs.mean(axis=0)
    mean /= (np.linalg.norm(mean) + 1e-8)

    top_texts = [slice_texts[i] for i in idx]
    keys = top_keywords(top_texts, k=keyword_k)
    return mean.astype("float32"), keys, top_texts


def extractive_answer(query: str, texts: List[str], max_sentences: int = 2) -> str:
    q = set(tokenize(query))
    scored = []
    for txt in texts:
        sents = re.split(r"(?<=[.!?])\s+", txt.strip())
        for s in sents:
            toks = set(tokenize(s))
            if not toks:
                continue
            overlap = len(q & toks) / (len(q) or 1)
            density = len(q & toks) / len(toks)
            scored.append((0.7 * overlap + 0.3 * density, s))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    seen = set()
    for _, sent in scored:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(sent)
        if len(out) >= max_sentences:
            break
    return " ".join(out) if out else ""


def resolve_workers(requested: int) -> int:
    return max(1, requested)


def profile_one_concept(
    concept: str,
    cvec: np.ndarray,
    parsed_slices: List[Tuple[str, int, int]],
    slice_texts: Dict[str, List[str]],
    slice_vecs: Dict[str, np.ndarray],
    top_m_docs: int,
    keyword_k: int,
    min_doc_support: int,
) -> Tuple[Dict, List[float], List[float], List[Dict]]:
    profiles = {}
    for sname, _, _ in parsed_slices:
        vec, keys, top_texts = concept_profile_for_slice(
            concept_vec=cvec,
            slice_texts=slice_texts[sname],
            slice_vecs=slice_vecs[sname],
            top_m_docs=top_m_docs,
            keyword_k=keyword_k,
        )
        profiles[sname] = {
            "vec": vec,
            "keywords": keys,
            "top_texts": top_texts,
        }

    transitions = []
    ordered = [s for s, _, _ in parsed_slices]
    sem_vals: List[float] = []
    nei_vals: List[float] = []
    valid_events: List[Dict] = []

    for s0, s1 in zip(ordered[:-1], ordered[1:]):
        v0 = profiles[s0]["vec"]
        v1 = profiles[s1]["vec"]
        semantic_shift = 1.0 - cosine(v0, v1)
        neigh_shift = jaccard_distance(set(profiles[s0]["keywords"]), set(profiles[s1]["keywords"]))
        support = min(len(profiles[s0]["top_texts"]), len(profiles[s1]["top_texts"]))
        confidence = max(0.0, min(1.0, support / float(max(min_doc_support, 1))))
        is_valid = support >= min_doc_support
        row = {
            "from_slice": s0,
            "to_slice": s1,
            "semantic_shift": round(float(semantic_shift), 6),
            "neighbor_shift": round(float(neigh_shift), 6),
            "confidence": round(float(confidence), 6),
            "support": int(support),
            "is_valid": bool(is_valid),
            "from_keywords": profiles[s0]["keywords"],
            "to_keywords": profiles[s1]["keywords"],
        }
        transitions.append(row)
        sem_vals.append(float(semantic_shift))
        nei_vals.append(float(neigh_shift))
        if is_valid:
            valid_events.append(
                {
                    "concept": concept,
                    "from_slice": s0,
                    "to_slice": s1,
                    "semantic_shift": row["semantic_shift"],
                    "neighbor_shift": row["neighbor_shift"],
                    "confidence": row["confidence"],
                    "support": row["support"],
                }
            )

    query_responses = []
    query_text = f"What are key ideas around {concept}?"
    for s in ordered:
        ans = extractive_answer(query_text, profiles[s]["top_texts"], max_sentences=2)
        query_responses.append(
            {
                "slice": s,
                "query": query_text,
                "answer": ans,
            }
        )

    return (
        {
            "concept": concept,
            "transitions": transitions,
            "query_responses": query_responses,
        },
        sem_vals,
        nei_vals,
        valid_events,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv concept drift analysis")
    parser.add_argument("--category", default="cs.CL")
    parser.add_argument("--query-all", default=None, help="Optional extra arXiv all: filter")
    parser.add_argument("--concepts", nargs="+", default=["transformer", "attention", "prompt", "alignment"])
    parser.add_argument(
        "--slices",
        nargs="+",
        default=["2017-2018", "2020-2021", "2024-2025"],
        help="Time slices as YYYY-YYYY",
    )
    parser.add_argument("--max-papers-per-slice", type=int, default=200)
    parser.add_argument("--top-m-docs", type=int, default=30)
    parser.add_argument("--keyword-k", type=int, default=25)
    parser.add_argument("--min-doc-support", type=int, default=10)
    parser.add_argument("--pause-s", type=float, default=0.8)
    parser.add_argument("--cache-dir", default="perf/cache/arxiv", help="Directory for fetched arXiv slice cache")
    parser.add_argument("--refresh-cache", action="store_true", help="Bypass cache and refetch slices")
    parser.add_argument("--refresh-emb-cache", action="store_true", help="Recompute and overwrite slice embedding cache")
    parser.add_argument("--embed-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    encoder = TextMiniLM()
    workers = resolve_workers(args.workers)

    # Load papers by slice
    slice_papers: Dict[str, List[Paper]] = {}
    parsed_slices = []
    for s in args.slices:
        a, b = s.split("-")
        y0, y1 = int(a), int(b)
        parsed_slices.append((s, y0, y1))

    for name, y0, y1 in parsed_slices:
        cache_dir = Path(args.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        key = cache_key(args.category, y0, y1, args.max_papers_per_slice, args.query_all)
        cache_path = cache_dir / f"{name}_{key}.json"

        papers: List[Paper]
        if cache_path.exists() and not args.refresh_cache:
            raw = json.loads(cache_path.read_text(encoding="utf-8"))
            papers = [Paper(**r) for r in raw]
            print(f"Loaded {name} from cache: {cache_path}")
        else:
            print(f"Fetching {name}...")
            papers = fetch_arxiv_slice(
                category=args.category,
                year_start=y0,
                year_end=y1,
                max_results=args.max_papers_per_slice,
                query_all=args.query_all,
                pause_s=args.pause_s,
            )
            cache_path.write_text(
                json.dumps([p.__dict__ for p in papers], indent=2),
                encoding="utf-8",
            )
            print(f"Cached {name}: {cache_path}")
        slice_papers[name] = papers
        print(f"  papers: {len(papers)}")

    # Pre-embed each slice once; reused for all concepts.
    slice_texts: Dict[str, List[str]] = {}
    slice_vecs: Dict[str, np.ndarray] = {}
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name, y0, y1 in parsed_slices:
        papers = slice_papers[name]
        key = cache_key(args.category, y0, y1, args.max_papers_per_slice, args.query_all)
        emb_path = cache_dir / f"{name}_{key}_emb.npz"
        texts = [f"{p.title}. {p.abstract}" for p in papers]

        loaded = False
        if emb_path.exists() and not args.refresh_emb_cache:
            try:
                emb_payload = np.load(emb_path, allow_pickle=True)
                cached_texts = [str(x) for x in emb_payload["texts"].tolist()]
                if cached_texts == texts:
                    vecs = emb_payload["vecs"].astype("float32")
                    slice_texts[name] = texts
                    slice_vecs[name] = vecs
                    loaded = True
                    print(f"Loaded embedding cache for {name}: {emb_path}")
            except Exception:
                loaded = False
        if not loaded:
            print(f"Encoding slice vectors for {name}...")
            vecs = encode_texts(texts, encoder=encoder, batch_size=args.embed_batch_size)
            np.savez_compressed(emb_path, texts=np.array(texts, dtype=object), vecs=vecs)
            slice_texts[name] = texts
            slice_vecs[name] = vecs
            print(f"Cached embeddings for {name}: {emb_path}")

    concept_results = []
    all_sem = []
    all_nei = []
    valid_events = []
    concept_vecs = encode_texts(args.concepts, encoder=encoder, batch_size=args.embed_batch_size)
    jobs = [(args.concepts[i], concept_vecs[i]) for i in range(len(args.concepts))]
    print(f"Profiling concepts with workers={workers}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                profile_one_concept,
                concept=concept,
                cvec=cvec,
                parsed_slices=parsed_slices,
                slice_texts=slice_texts,
                slice_vecs=slice_vecs,
                top_m_docs=args.top_m_docs,
                keyword_k=args.keyword_k,
                min_doc_support=args.min_doc_support,
            )
            for concept, cvec in jobs
        ]
        for fut in futs:
            result, sem_vals, nei_vals, valid = fut.result()
            concept_results.append(result)
            all_sem.extend(sem_vals)
            all_nei.extend(nei_vals)
            valid_events.extend(valid)

    concept_results.sort(key=lambda x: x["concept"])

    canonical_cfg = {
        "source": "arXiv API",
        "category": args.category,
        "query_all": args.query_all,
        "slices": args.slices,
        "max_papers_per_slice": args.max_papers_per_slice,
        "top_m_docs": args.top_m_docs,
        "keyword_k": args.keyword_k,
        "min_doc_support": args.min_doc_support,
        "concepts": args.concepts,
    }
    cfg_hash = hashlib.sha1(json.dumps(canonical_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    payload = {
        "schema_version": "drift.v1",
        "benchmark_name": "arxiv_concept_drift",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "source": "arXiv API",
        "config": {
            "category": args.category,
            "query_all": args.query_all,
            "slices": args.slices,
            "max_papers_per_slice": args.max_papers_per_slice,
            "top_m_docs": args.top_m_docs,
            "keyword_k": args.keyword_k,
            "min_doc_support": args.min_doc_support,
            "workers": workers,
            "concepts": args.concepts,
        },
        "slice_stats": {
            s: {"papers": len(ps)} for s, ps in slice_papers.items()
        },
        "summary": {
            "avg_semantic_shift": round(float(statistics.mean(all_sem)), 6) if all_sem else 0.0,
            "avg_neighbor_shift": round(float(statistics.mean(all_nei)), 6) if all_nei else 0.0,
            "num_concepts": len(args.concepts),
            "num_transitions": len(all_sem),
            "num_transitions_valid": len(valid_events),
        },
        "drift_events": sorted(valid_events, key=lambda x: x["semantic_shift"], reverse=True),
        "concepts": concept_results,
    }

    out = Path(args.output) if args.output else Path(
        f"perf/arxiv_concept_drift_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nConcept drift analysis complete")
    print(f"Output: {out}")
    print(f"Avg semantic shift: {payload['summary']['avg_semantic_shift']:.4f}")
    print(f"Avg neighbor shift: {payload['summary']['avg_neighbor_shift']:.4f}")


if __name__ == "__main__":
    main()
