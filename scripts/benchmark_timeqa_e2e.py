#!/usr/bin/env python3
"""End-to-end TimeQA benchmark for Hologram."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from datasets import load_dataset

from hologram.api import Hologram
from hologram.chatbot import OpenAIChatProvider
from hologram.glyphs import GlyphRegistry
from hologram.gravity import GravityField
from hologram.glyph_router import GlyphRouter
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


@dataclass
class SampleResult:
    sample_id: str
    level: str
    question: str
    answer: str
    targets: List[str]
    token_f1: float
    exact_match: float
    target_hit: float
    groundedness_proxy: float
    latency_ms: float


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def _keyword_set(text: str) -> set[str]:
    return {t for t in _tokenize(text) if t not in STOPWORDS}


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


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


def groundedness_proxy(answer: str, contexts: Sequence[str]) -> float:
    ans = _keyword_set(answer)
    if not ans:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx.update(_keyword_set(c))
    return _safe_div(len(ans & ctx), len(ans))


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


def generate_openai(provider: OpenAIChatProvider, question: str, contexts: Sequence[str]) -> str:
    context_block = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "Answer the question using only the retrieved documents.\n"
        "Output exactly one line in this format:\n"
        "Final answer: <very short answer>\n"
        "If the answer is not in context, output:\n"
        "Final answer: Unknown from provided context.\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Documents:\n{context_block}\n\n"
        "Answer:"
    )
    raw = provider.generate([{"role": "user", "content": prompt}]).strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    if first_line.lower().startswith("final answer:"):
        return first_line[len("Final answer:"):].strip()
    return raw


def reset_hologram(hg: Hologram, use_gravity: bool) -> None:
    vec_dim = hg.store.vec_dim
    hg.store = MemoryStore(vec_dim=vec_dim)
    hg.glyphs = GlyphRegistry(hg.store)
    hg.manifold = LatentManifold(dim=vec_dim)
    hg.field = GravityField(dim=vec_dim) if use_gravity else None
    hg.router = GlyphRouter(hg.store, hg.glyphs, gravity_field=hg.field)


def run_sample(
    row: Dict,
    *,
    hg: Hologram,
    top_k: int,
    use_drift: bool,
    use_gravity: bool,
    generator: str,
    gen_provider: OpenAIChatProvider | None,
) -> SampleResult:
    reset_hologram(hg, use_gravity=use_gravity)
    sample_id = str(row.get("idx", ""))
    question = str(row.get("question", ""))
    level = str(row.get("level", "unknown"))
    targets = [str(t) for t in (row.get("targets") or []) if str(t).strip()]
    context = str(row.get("context", ""))

    glyph = f"timeqa:{sample_id}"
    hg.glyphs.create(glyph, title=f"TimeQA {sample_id}")

    chunks = chunk_text(context, sents_per_chunk=3)
    for i, chunk in enumerate(chunks, start=1):
        hg.add_text(
            glyph_id=glyph,
            text=chunk,
            trace_id=f"doc:{i}",
            do_extract_concepts=False,
            add_to_field=use_gravity,
            skip_nlp=True,
            origin="benchmark",
        )

    t0 = time.time()
    if use_drift:
        packet = hg.search_with_drift(question, top_k_traces=top_k)
        hits = [(r["trace"], float(r["score"])) for r in packet["results"]]
    else:
        hits = hg.search_text(question, top_k=top_k)
    latency_ms = (time.time() - t0) * 1000

    contexts = [tr.content for tr, _ in hits]
    if generator == "openai":
        if gen_provider is None:
            raise RuntimeError("OPENAI_API_KEY is required for --generator openai")
        answer = generate_openai(gen_provider, question, contexts)
    else:
        answer = generate_extractive(question, contexts)

    best_f1 = max((token_f1(answer, t) for t in targets), default=0.0)
    best_em = max((exact_match(answer, t) for t in targets), default=0.0)
    hit = 0.0
    norm_answer = " ".join(_tokenize(answer))
    for t in targets:
        norm_t = " ".join(_tokenize(t))
        if norm_t and norm_t in norm_answer:
            hit = 1.0
            break

    return SampleResult(
        sample_id=sample_id,
        level=level,
        question=question,
        answer=answer,
        targets=targets,
        token_f1=best_f1,
        exact_match=best_em,
        target_hit=hit,
        groundedness_proxy=groundedness_proxy(answer, contexts),
        latency_ms=latency_ms,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end TimeQA benchmark for Hologram")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--shuffle", action="store_true", help="Shuffle before taking --limit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --shuffle")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    parser.add_argument("--gen-model", default="gpt-4o-mini")
    parser.add_argument("--use-gravity", action="store_true")
    parser.add_argument("--use-drift", action="store_true")
    parser.add_argument("--hf-home", default="/tmp/hf_home")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if load_dotenv is not None:
        load_dotenv()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    print(f"Loading TimeQA split={args.split}")
    ds = load_dataset("hugosousa/TimeQA", split=args.split)
    if args.shuffle:
        ds = ds.shuffle(seed=args.seed)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    gen_provider = None
    if args.generator == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --generator openai")
        gen_provider = OpenAIChatProvider(api_key=api_key, model=args.gen_model)

    hg = Hologram.init(
        encoder_mode="minilm",
        use_clip=False,
        use_gravity=args.use_gravity,
        auto_ingest_system=False,
    )

    rows: List[SampleResult] = []
    start_all = time.time()
    for idx, row in enumerate(ds, start=1):
        rows.append(
            run_sample(
                row,
                hg=hg,
                top_k=args.top_k,
                use_drift=args.use_drift,
                use_gravity=args.use_gravity,
                generator=args.generator,
                gen_provider=gen_provider,
            )
        )
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(ds)}")
    elapsed = time.time() - start_all

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "TimeQA-E2E",
        "dataset": "hugosousa/TimeQA",
        "split": args.split,
        "limit": len(rows),
        "config": {
            "top_k": args.top_k,
            "use_gravity": args.use_gravity,
            "use_drift": args.use_drift,
            "generator": args.generator,
            "gen_model": args.gen_model if args.generator == "openai" else None,
        },
        "timing": {
            "total_seconds": elapsed,
            "seconds_per_sample": _safe_div(elapsed, len(rows)),
            "retrieval_latency_ms": statistics.mean([r.latency_ms for r in rows]) if rows else 0.0,
        },
        "metrics": {
            "token_f1_vs_targets": statistics.mean([r.token_f1 for r in rows]) if rows else 0.0,
            "exact_match_vs_targets": statistics.mean([r.exact_match for r in rows]) if rows else 0.0,
            "target_hit_rate": statistics.mean([r.target_hit for r in rows]) if rows else 0.0,
            "groundedness_proxy": statistics.mean([r.groundedness_proxy for r in rows]) if rows else 0.0,
        },
        "by_level": {},
        "samples": [
            {
                "id": r.sample_id,
                "level": r.level,
                "question": r.question,
                "answer": r.answer,
                "targets": r.targets,
                "token_f1": r.token_f1,
                "exact_match": r.exact_match,
                "target_hit": r.target_hit,
                "groundedness_proxy": r.groundedness_proxy,
                "latency_ms": r.latency_ms,
            }
            for r in rows
        ],
    }

    levels = sorted({r.level for r in rows})
    for lv in levels:
        subset = [r for r in rows if r.level == lv]
        summary["by_level"][lv] = {
            "count": len(subset),
            "token_f1_vs_targets": statistics.mean([r.token_f1 for r in subset]) if subset else 0.0,
            "exact_match_vs_targets": statistics.mean([r.exact_match for r in subset]) if subset else 0.0,
            "target_hit_rate": statistics.mean([r.target_hit for r in subset]) if subset else 0.0,
        }

    out = Path(args.output) if args.output else Path(
        f"perf/timeqa_e2e_{args.split}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBenchmark complete")
    print(f"Output: {out}")
    print(f"Token-F1(targets): {summary['metrics']['token_f1_vs_targets']:.4f}")
    print(f"EM(targets):       {summary['metrics']['exact_match_vs_targets']:.4f}")
    print(f"Target hit rate:   {summary['metrics']['target_hit_rate']:.4f}")


if __name__ == "__main__":
    main()
