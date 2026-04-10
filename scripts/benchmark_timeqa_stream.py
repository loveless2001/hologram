#!/usr/bin/env python3
"""Streaming temporal benchmark for TimeQA with persistent Hologram state."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset

from hologram.api import Hologram
from hologram.glyphs import GlyphRegistry
from hologram.gravity import GravityField
from hologram.manifold import LatentManifold
from hologram.store import MemoryStore

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "it",
    "that", "this", "as", "from", "about", "into", "over", "after", "before",
    "than", "then", "which", "who", "whom", "what", "when", "where", "why", "how",
}
YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


@dataclass
class EvalRow:
    idx: str
    level: str
    question: str
    targets: List[str]
    context: str
    anchor_year: int
    seq: int


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _keyword_set(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in STOPWORDS}


def token_f1(pred: str, ref: str) -> float:
    p = _tokenize(pred)
    r = _tokenize(ref)
    if not p or not r:
        return 0.0
    p_counts: Dict[str, int] = {}
    r_counts: Dict[str, int] = {}
    for tok in p:
        p_counts[tok] = p_counts.get(tok, 0) + 1
    for tok in r:
        r_counts[tok] = r_counts.get(tok, 0) + 1
    common = 0
    for tok, cnt in p_counts.items():
        common += min(cnt, r_counts.get(tok, 0))
    precision = _safe_div(common, len(p))
    recall = _safe_div(common, len(r))
    return _safe_div(2 * precision * recall, precision + recall)


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if " ".join(_tokenize(pred)) == " ".join(_tokenize(ref)) else 0.0


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, sents_per_chunk: int = 3) -> List[str]:
    sents = split_sentences(text)
    if not sents:
        return [text[:800]] if text else []
    chunks = []
    for i in range(0, len(sents), sents_per_chunk):
        chunks.append(" ".join(sents[i : i + sents_per_chunk]))
    return chunks


def generate_extractive(question: str, contexts: Sequence[str], max_sentences: int = 2) -> str:
    q_terms = _keyword_set(question)
    scored: List[Tuple[float, str]] = []
    for ctx in contexts:
        for sent in split_sentences(ctx):
            s_terms = _keyword_set(sent)
            if not s_terms:
                continue
            overlap = _safe_div(len(q_terms & s_terms), len(q_terms) if q_terms else 1)
            density = _safe_div(len(q_terms & s_terms), len(s_terms))
            score = 0.7 * overlap + 0.3 * density
            scored.append((score, sent))
    scored.sort(key=lambda x: x[0], reverse=True)
    chosen: List[str] = []
    seen = set()
    for _, sent in scored:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sent)
        if len(chosen) >= max_sentences:
            break
    if chosen:
        return " ".join(chosen)
    return contexts[0][:300] if contexts else "Unknown from provided context."


def extract_anchor_year(question: str, context: str) -> int:
    years_q = [int(y) for y in YEAR_RE.findall(question)]
    if years_q:
        return max(years_q)
    years_c = [int(y) for y in YEAR_RE.findall(context)]
    if years_c:
        return min(years_c)
    return 1900


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


def best_scores(answer: str, targets: Sequence[str]) -> Tuple[float, float, float]:
    if not targets:
        return 0.0, 0.0, 0.0
    f1 = max(token_f1(answer, t) for t in targets)
    em = max(exact_match(answer, t) for t in targets)
    norm_answer = " ".join(_tokenize(answer))
    hit = 0.0
    for t in targets:
        norm_t = " ".join(_tokenize(t))
        if norm_t and norm_t in norm_answer:
            hit = 1.0
            break
    return f1, em, hit


def maybe_retrieve(hg: Hologram, question: str, top_k: int, use_drift: bool) -> Tuple[List[str], float]:
    t0 = time.time()
    if use_drift:
        packet = hg.search_with_drift(question, top_k_traces=top_k)
        hits = [(r["trace"], float(r["score"])) for r in packet["results"]]
    else:
        hits = hg.search_text(question, top_k=top_k)
    latency_ms = (time.time() - t0) * 1000
    return [tr.content for tr, _ in hits], latency_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming TimeQA benchmark for persistent temporal memory")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-gravity", action="store_true")
    parser.add_argument("--use-drift", action="store_true")
    parser.add_argument("--shuffle-entities", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--replay-every", type=int, default=20)
    parser.add_argument("--replay-size", type=int, default=20)
    parser.add_argument("--hf-home", default="/tmp/hf_home")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()
    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    ds = load_dataset("hugosousa/TimeQA", split=args.split)
    rows: List[EvalRow] = []
    for row in ds:
        idx = str(row.get("idx", ""))
        level = str(row.get("level", "unknown"))
        question = str(row.get("question", ""))
        targets = [str(t) for t in (row.get("targets") or []) if str(t).strip()]
        context = str(row.get("context", ""))
        _, seq = parse_idx(idx)
        rows.append(
            EvalRow(
                idx=idx,
                level=level,
                question=question,
                targets=targets,
                context=context,
                anchor_year=extract_anchor_year(question, context),
                seq=seq,
            )
        )

    grouped: Dict[str, List[EvalRow]] = {}
    for r in rows:
        entity, _ = parse_idx(r.idx)
        grouped.setdefault(entity, []).append(r)
    for entity in grouped:
        grouped[entity].sort(key=lambda x: (x.anchor_year, x.seq))

    entities = list(grouped.keys())
    entities.sort(key=lambda e: min(x.anchor_year for x in grouped[e]))
    if args.shuffle_entities:
        random.Random(args.seed).shuffle(entities)

    stream: List[EvalRow] = []
    for e in entities:
        stream.extend(grouped[e])
    if args.limit:
        stream = stream[: min(args.limit, len(stream))]

    hg = Hologram.init(
        encoder_mode="minilm",
        use_clip=False,
        use_gravity=args.use_gravity,
        auto_ingest_system=False,
    )
    # Ensure clean persistent state.
    vec_dim = hg.store.vec_dim
    hg.store = MemoryStore(vec_dim=vec_dim)
    hg.glyphs = GlyphRegistry(hg.store)
    hg.manifold = LatentManifold(dim=vec_dim)
    hg.field = GravityField(dim=vec_dim) if args.use_gravity else None

    forward_f1: List[float] = []
    forward_em: List[float] = []
    forward_hit: List[float] = []
    forward_ms: List[float] = []
    replay_f1: List[float] = []
    replay_em: List[float] = []
    replay_hit: List[float] = []
    replay_ms: List[float] = []
    seen: List[EvalRow] = []

    t_all = time.time()
    for i, row in enumerate(stream, start=1):
        glyph_id = f"timeqa_stream:{row.idx}"
        hg.glyphs.create(glyph_id, title=f"TimeQA {row.idx}")
        chunks = chunk_text(row.context, sents_per_chunk=3)
        for j, chunk in enumerate(chunks, start=1):
            hg.add_text(
                glyph_id=glyph_id,
                text=chunk,
                trace_id=f"{row.idx}:doc:{j}",
                do_extract_concepts=False,
                add_to_field=args.use_gravity,
                skip_nlp=True,
                origin="benchmark",
            )

        contexts, latency = maybe_retrieve(hg, row.question, args.top_k, args.use_drift)
        ans = generate_extractive(row.question, contexts)
        f1, em, hit = best_scores(ans, row.targets)
        forward_f1.append(f1)
        forward_em.append(em)
        forward_hit.append(hit)
        forward_ms.append(latency)
        seen.append(row)

        if args.replay_every > 0 and i % args.replay_every == 0 and seen:
            sample_n = min(args.replay_size, len(seen))
            sample = random.Random(args.seed + i).sample(seen, sample_n)
            for old in sample:
                rctx, rlat = maybe_retrieve(hg, old.question, args.top_k, args.use_drift)
                rans = generate_extractive(old.question, rctx)
                rf1, rem, rhit = best_scores(rans, old.targets)
                replay_f1.append(rf1)
                replay_em.append(rem)
                replay_hit.append(rhit)
                replay_ms.append(rlat)

        if i % 20 == 0:
            print(f"Processed {i}/{len(stream)}")

    elapsed = time.time() - t_all
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "TimeQA-Stream",
        "dataset": "hugosousa/TimeQA",
        "split": args.split,
        "limit": len(stream),
        "config": {
            "top_k": args.top_k,
            "use_gravity": args.use_gravity,
            "use_drift": args.use_drift,
            "shuffle_entities": args.shuffle_entities,
            "seed": args.seed,
            "replay_every": args.replay_every,
            "replay_size": args.replay_size,
            "generator": "extractive",
        },
        "timing": {
            "total_seconds": elapsed,
            "seconds_per_sample": _safe_div(elapsed, len(stream)),
            "forward_retrieval_ms": statistics.mean(forward_ms) if forward_ms else 0.0,
            "replay_retrieval_ms": statistics.mean(replay_ms) if replay_ms else 0.0,
        },
        "metrics": {
            "forward_token_f1": statistics.mean(forward_f1) if forward_f1 else 0.0,
            "forward_exact_match": statistics.mean(forward_em) if forward_em else 0.0,
            "forward_target_hit": statistics.mean(forward_hit) if forward_hit else 0.0,
            "replay_token_f1": statistics.mean(replay_f1) if replay_f1 else 0.0,
            "replay_exact_match": statistics.mean(replay_em) if replay_em else 0.0,
            "replay_target_hit": statistics.mean(replay_hit) if replay_hit else 0.0,
            "replay_eval_count": len(replay_f1),
        },
    }

    out = Path(args.output) if args.output else Path(
        f"perf/timeqa_stream_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBenchmark complete")
    print(f"Output: {out}")
    print(f"Forward F1: {payload['metrics']['forward_token_f1']:.4f}")
    print(f"Replay  F1: {payload['metrics']['replay_token_f1']:.4f}")
    print(f"Forward hit: {payload['metrics']['forward_target_hit']:.4f}")
    print(f"Replay  hit: {payload['metrics']['replay_target_hit']:.4f}")


if __name__ == "__main__":
    main()
