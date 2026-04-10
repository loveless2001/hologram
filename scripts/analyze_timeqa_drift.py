#!/usr/bin/env python3
"""Analyze concept drift over time on TimeQA.

This script builds year-sliced entity representations and quantifies:
1) Semantic drift: cosine distance between an entity's vectors across years.
2) Neighborhood drift: Jaccard change in top-k nearest entity neighbors by year.

Output is a JSON report suitable for longitudinal analysis.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import load_dataset

from hologram.embeddings import TextMiniLM
from hologram.manifold import LatentManifold

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


@dataclass
class Row:
    idx: str
    entity: str
    year: int
    text: str


def parse_idx(idx: str) -> Tuple[str, int]:
    parts = idx.split("#")
    entity = parts[0] if parts else idx
    seq = 0
    if parts:
        try:
            seq = int(parts[-1])
        except Exception:
            seq = 0
    return entity, seq


def extract_anchor_year(question: str, context: str) -> int:
    years_q = [int(y) for y in YEAR_RE.findall(question)]
    if years_q:
        return max(years_q)
    years_c = [int(y) for y in YEAR_RE.findall(context)]
    if years_c:
        return min(years_c)
    return 1900


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a) + 1e-8
    bn = np.linalg.norm(b) + 1e-8
    return float(np.dot(a, b) / (an * bn))


def jaccard_distance(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))


def top_k_neighbors(
    vectors: Dict[str, np.ndarray],
    entity: str,
    k: int,
) -> List[str]:
    if entity not in vectors:
        return []
    v = vectors[entity]
    scores: List[Tuple[str, float]] = []
    for other, ov in vectors.items():
        if other == entity:
            continue
        scores.append((other, cosine(v, ov)))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [name for name, _ in scores[:k]]


def as_float(x: float) -> float:
    return float(round(x, 6))


def row_key(row: Row) -> str:
    h = hashlib.sha1(row.text.encode("utf-8")).hexdigest()[:12]
    return f"{row.idx}|{row.year}|{h}"


def resolve_workers(requested: int) -> int:
    return max(1, requested)


def build_year_neighbors(
    year: int,
    entities: Dict[str, np.ndarray],
    top_k: int,
) -> Tuple[int, Dict[str, List[str]]]:
    out = {entity: top_k_neighbors(entities, entity, top_k) for entity in entities}
    return year, out


def compute_entity_transitions(
    entity: str,
    years: List[int],
    year_entity_mean: Dict[int, Dict[str, np.ndarray]],
    year_neighbor_index: Dict[int, Dict[str, List[str]]],
    min_neighbor_support: int,
) -> Tuple[List[Dict], List[Dict], List[float], List[float]]:
    transitions = []
    rows = []
    semantic_shifts = []
    neighbor_shifts = []
    for prev_year, cur_year in zip(years[:-1], years[1:]):
        prev_vec = year_entity_mean[prev_year][entity]
        cur_vec = year_entity_mean[cur_year][entity]
        semantic_shift = 1.0 - cosine(prev_vec, cur_vec)

        prev_neighbors = set(year_neighbor_index.get(prev_year, {}).get(entity, []))
        cur_neighbors = set(year_neighbor_index.get(cur_year, {}).get(entity, []))
        neigh_shift = jaccard_distance(prev_neighbors, cur_neighbors)
        support = min(len(prev_neighbors), len(cur_neighbors))
        confidence = max(0.0, min(1.0, support / float(max(min_neighbor_support, 1))))
        is_valid = support >= min_neighbor_support
        transition = {
            "from_year": int(prev_year),
            "to_year": int(cur_year),
            "semantic_shift": as_float(semantic_shift),
            "neighbor_shift": as_float(neigh_shift),
            "confidence": as_float(confidence),
            "support": int(support),
            "is_valid": bool(is_valid),
            "prev_neighbors": sorted(prev_neighbors),
            "cur_neighbors": sorted(cur_neighbors),
        }
        transitions.append(transition)
        row = {
            "entity": entity,
            "from_year": int(prev_year),
            "to_year": int(cur_year),
            "semantic_shift": as_float(semantic_shift),
            "neighbor_shift": as_float(neigh_shift),
            "confidence": as_float(confidence),
            "support": int(support),
            "is_valid": bool(is_valid),
        }
        rows.append(row)
        semantic_shifts.append(semantic_shift)
        neighbor_shifts.append(neigh_shift)
    return transitions, rows, semantic_shifts, neighbor_shifts


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze TimeQA concept drift by year")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=2000, help="Max samples to load after sorting by year")
    parser.add_argument("--top-k-neighbors", type=int, default=8)
    parser.add_argument("--min-years-per-entity", type=int, default=2)
    parser.add_argument("--min-entities-per-year", type=int, default=2)
    parser.add_argument("--min-neighbor-support", type=int, default=1)
    parser.add_argument(
        "--text-source",
        default="both",
        choices=["context", "question", "both"],
        help="Text used to encode each sample vector",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1200,
        help="Truncate context to this many chars before encoding",
    )
    parser.add_argument("--hf-home", default="/tmp/hf_home")
    parser.add_argument("--emb-cache", default=None, help="Optional .npz path for embedding cache")
    parser.add_argument("--no-emb-cache", action="store_true", help="Disable reading/writing embedding cache")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    ds = load_dataset("hugosousa/TimeQA", split=args.split)

    rows: List[Row] = []
    for item in ds:
        idx = str(item.get("idx", ""))
        question = str(item.get("question", ""))
        context = str(item.get("context", ""))
        entity, _ = parse_idx(idx)
        year = extract_anchor_year(question, context)

        trimmed_context = context[: args.max_context_chars] if args.max_context_chars > 0 else context
        if args.text_source == "context":
            text = trimmed_context
        elif args.text_source == "question":
            text = question
        else:
            text = f"{question}\n{trimmed_context}"

        rows.append(Row(idx=idx, entity=entity, year=year, text=text))

    rows.sort(key=lambda r: (r.year, r.entity, r.idx))
    if args.limit > 0:
        rows = rows[: min(args.limit, len(rows))]

    encoder = TextMiniLM()
    manifold = LatentManifold(dim=384)
    workers = resolve_workers(args.workers)
    use_cache = not args.no_emb_cache
    if args.emb_cache:
        emb_cache = Path(args.emb_cache)
    else:
        emb_cache = Path(
            f"perf/cache/timeqa_emb_{args.split}_{len(rows)}_{args.text_source}_{args.max_context_chars}.npz"
        )
    emb_cache.parent.mkdir(parents=True, exist_ok=True)

    keys = [row_key(r) for r in rows]
    vectors = None
    if use_cache and emb_cache.exists():
        try:
            payload = np.load(emb_cache, allow_pickle=True)
            cached_keys = [str(x) for x in payload["keys"].tolist()]
            if cached_keys == keys:
                vectors = payload["vecs"].astype("float32")
                print(f"Loaded embedding cache: {emb_cache}")
        except Exception:
            vectors = None

    # Aggregate vectors by (year, entity)
    year_entity_vectors: Dict[int, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    encoded_vecs: List[np.ndarray] = []
    for i, row in enumerate(rows, start=1):
        if vectors is not None:
            vec = vectors[i - 1]
        else:
            vec = manifold.align_text(row.text, encoder)
        encoded_vecs.append(vec)
        year_entity_vectors[row.year][row.entity].append(vec)
        if i % 200 == 0:
            print(f"Encoded {i}/{len(rows)} samples")

    if vectors is None and use_cache:
        vec_array = np.stack(encoded_vecs).astype("float32")
        np.savez_compressed(emb_cache, keys=np.array(keys, dtype=object), vecs=vec_array)
        print(f"Saved embedding cache: {emb_cache}")

    # Mean vector per (year, entity)
    year_entity_mean: Dict[int, Dict[str, np.ndarray]] = {}
    for year, entities in year_entity_vectors.items():
        year_entity_mean[year] = {}
        for entity, vecs in entities.items():
            m = np.mean(np.stack(vecs), axis=0).astype("float32")
            n = np.linalg.norm(m) + 1e-8
            year_entity_mean[year][entity] = m / n

    # Build temporal nearest-neighbor index once per year.
    year_neighbor_index: Dict[int, Dict[str, List[str]]] = {}
    jobs = [
        (year, entities)
        for year, entities in year_entity_mean.items()
        if len(entities) >= args.min_entities_per_year
    ]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(build_year_neighbors, y, e, args.top_k_neighbors) for y, e in jobs]
        for fut in futs:
            year, idx = fut.result()
            year_neighbor_index[year] = idx

    # Build per-entity timeline
    entity_years: Dict[str, List[int]] = defaultdict(list)
    for year, entities in year_entity_mean.items():
        for entity in entities:
            entity_years[entity].append(year)
    for entity in entity_years:
        entity_years[entity] = sorted(set(entity_years[entity]))

    per_entity = []
    all_transition_rows = []
    valid_transition_rows = []
    semantic_shifts = []
    neighbor_shifts = []

    entity_jobs = [(entity, years) for entity, years in entity_years.items() if len(years) >= args.min_years_per_entity]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(
                compute_entity_transitions,
                entity,
                years,
                year_entity_mean,
                year_neighbor_index,
                args.min_neighbor_support,
            )
            for entity, years in entity_jobs
        ]
        for (entity, years), fut in zip(entity_jobs, futs):
            transitions, rows_out, sem_vals, nei_vals = fut.result()
            if not transitions:
                continue
            semantic_shifts.extend(sem_vals)
            neighbor_shifts.extend(nei_vals)
            all_transition_rows.extend(rows_out)
            valid_transition_rows.extend([r for r in rows_out if r["is_valid"]])

            avg_sem = statistics.mean(t["semantic_shift"] for t in transitions)
            avg_nei = statistics.mean(t["neighbor_shift"] for t in transitions)
            per_entity.append(
                {
                    "entity": entity,
                    "years": years,
                    "num_years": len(years),
                    "num_transitions": len(transitions),
                    "avg_semantic_shift": as_float(avg_sem),
                    "avg_neighbor_shift": as_float(avg_nei),
                    "max_semantic_shift": as_float(max(t["semantic_shift"] for t in transitions)),
                    "max_neighbor_shift": as_float(max(t["neighbor_shift"] for t in transitions)),
                    "transitions": transitions,
                }
            )

    per_entity.sort(key=lambda x: x["avg_semantic_shift"], reverse=True)
    all_transition_rows.sort(key=lambda x: x["semantic_shift"], reverse=True)

    years_all = sorted(year_entity_mean.keys())
    semantic_shifts_sorted = sorted(semantic_shifts)
    p90_idx = int(0.9 * (len(semantic_shifts_sorted) - 1)) if semantic_shifts_sorted else 0
    p90 = semantic_shifts_sorted[p90_idx] if semantic_shifts_sorted else 0.0

    canonical_cfg = {
        "dataset": "hugosousa/TimeQA",
        "split": args.split,
        "limit": args.limit,
        "top_k_neighbors": args.top_k_neighbors,
        "min_years_per_entity": args.min_years_per_entity,
        "min_entities_per_year": args.min_entities_per_year,
        "min_neighbor_support": args.min_neighbor_support,
        "text_source": args.text_source,
        "max_context_chars": args.max_context_chars,
    }
    cfg_hash = hashlib.sha1(json.dumps(canonical_cfg, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    payload = {
        "schema_version": "drift.v1",
        "benchmark_name": "timeqa_concept_drift",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": cfg_hash,
        "dataset": "hugosousa/TimeQA",
        "split": args.split,
        "config": {
            "limit": args.limit,
            "top_k_neighbors": args.top_k_neighbors,
            "min_years_per_entity": args.min_years_per_entity,
            "min_entities_per_year": args.min_entities_per_year,
            "min_neighbor_support": args.min_neighbor_support,
            "workers": workers,
            "text_source": args.text_source,
            "max_context_chars": args.max_context_chars,
        },
        "summary": {
            "num_rows": len(rows),
            "num_years": len(years_all),
            "year_min": int(min(years_all)) if years_all else None,
            "year_max": int(max(years_all)) if years_all else None,
            "num_entities_total": len(entity_years),
            "num_entities_analyzed": len(per_entity),
            "num_transitions_total": len(all_transition_rows),
            "num_transitions_valid": len(valid_transition_rows),
            "avg_semantic_shift": as_float(statistics.mean(semantic_shifts)) if semantic_shifts else 0.0,
            "median_semantic_shift": as_float(statistics.median(semantic_shifts)) if semantic_shifts else 0.0,
            "p90_semantic_shift": as_float(p90),
            "avg_neighbor_shift": as_float(statistics.mean(neighbor_shifts)) if neighbor_shifts else 0.0,
        },
        "top_entities_by_avg_semantic_shift": per_entity[:25],
        "top_transitions_by_semantic_shift": sorted(
            valid_transition_rows if valid_transition_rows else all_transition_rows,
            key=lambda x: x["semantic_shift"],
            reverse=True,
        )[:100],
        "drift_events": sorted(
            valid_transition_rows if valid_transition_rows else all_transition_rows,
            key=lambda x: (x["semantic_shift"], x.get("confidence", 0.0)),
            reverse=True,
        )[:250],
        "temporal_index": {
            "years_indexed": len(year_neighbor_index),
            "entities_per_year_avg": as_float(
                statistics.mean(len(v) for v in year_neighbor_index.values())
            ) if year_neighbor_index else 0.0,
        },
        "entities": per_entity,
    }

    out = Path(args.output) if args.output else Path(
        f"perf/timeqa_drift_analysis_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nDrift analysis complete")
    print(f"Output: {out}")
    print(f"Entities analyzed: {payload['summary']['num_entities_analyzed']}")
    print(f"Transitions: {payload['summary']['num_transitions_total']}")
    print(f"Avg semantic shift: {payload['summary']['avg_semantic_shift']:.4f}")
    print(f"Avg neighbor shift: {payload['summary']['avg_neighbor_shift']:.4f}")


if __name__ == "__main__":
    main()
