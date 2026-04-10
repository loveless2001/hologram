#!/usr/bin/env python3
"""End-to-end RAGBench-style QA benchmark for Hologram.

This script runs a full RAG loop on a subset of the public RAGBench dataset:
1) Ingest provided documents into Hologram.
2) Retrieve top-k context for each question.
3) Generate an answer (extractive baseline or OpenAI model).
4) Score answers with reference/proxy metrics, and optional LLM-judge metrics.
"""

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
from typing import Dict, List, Optional, Sequence, Tuple

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

try:
    from sentence_transformers import CrossEncoder
except Exception:  # pragma: no cover
    CrossEncoder = None


STOPWORDS = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for",
    "with", "by", "is", "are", "was", "were", "be", "been", "being", "it",
    "that", "this", "as", "from", "about", "into", "over", "after", "before",
    "than", "then", "which", "who", "whom", "what", "when", "where", "why", "how",
}


@dataclass
class SampleResult:
    sample_id: str
    question: str
    answer: str
    reference: str
    token_f1: float
    exact_match: float
    groundedness_proxy: float
    question_coverage: float
    retrieval_doc_recall: float
    latency_ms: float


class PairReranker:
    """Optional cross-encoder reranker over (question, context) pairs."""

    def __init__(self, model_name: str):
        if CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is required for reranking."
            )
        self.model = CrossEncoder(model_name)

    def rerank(self, question: str, contexts: Sequence[str]) -> List[Tuple[float, str]]:
        if not contexts:
            return []
        pairs = [[question, c] for c in contexts]
        scores = self.model.predict(pairs)
        ranked = list(zip([float(s) for s in scores], contexts))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked


def reset_hologram(hg: Hologram, use_gravity: bool) -> None:
    """Reuse loaded encoders but reset memory/index state for next sample."""
    vec_dim = hg.store.vec_dim
    hg.store = MemoryStore(vec_dim=vec_dim)
    hg.glyphs = GlyphRegistry(hg.store)
    hg.manifold = LatentManifold(dim=vec_dim)
    hg.field = GravityField(dim=vec_dim) if use_gravity else None
    hg.router = GlyphRouter(hg.store, hg.glyphs, gravity_field=hg.field)


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
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, ref: str) -> float:
    norm_pred = " ".join(_tokenize(pred))
    norm_ref = " ".join(_tokenize(ref))
    return 1.0 if norm_pred == norm_ref else 0.0


def groundedness_proxy(answer: str, contexts: Sequence[str]) -> float:
    ans = _keyword_set(answer)
    if not ans:
        return 0.0
    ctx = set()
    for c in contexts:
        ctx.update(_keyword_set(c))
    return _safe_div(len(ans & ctx), len(ans))


def question_coverage(answer: str, question: str) -> float:
    q = _keyword_set(question)
    if not q:
        return 0.0
    a = _keyword_set(answer)
    return _safe_div(len(q & a), len(q))


def _extract_relevant_doc_indices(keys: Sequence[str]) -> set[int]:
    indices = set()
    for k in keys:
        m = re.match(r"^(\d+)", str(k))
        if m:
            indices.add(int(m.group(1)))
    return indices


def retrieval_doc_recall(retrieved_doc_ids: Sequence[int], relevant_keys: Sequence[str]) -> float:
    gold = _extract_relevant_doc_indices(relevant_keys)
    if not gold:
        return 0.0
    got = set(retrieved_doc_ids)
    return _safe_div(len(gold & got), len(gold))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


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
    return contexts[0][:300] if contexts else "I do not have enough evidence in the retrieved context."


def generate_openai(
    provider: OpenAIChatProvider,
    question: str,
    contexts: Sequence[str],
    prompt_style: str = "strict",
) -> str:
    context_block = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(contexts))
    if prompt_style == "concise_evidence":
        prompt = (
            "Answer using only the retrieved documents.\n"
            "Format exactly:\n"
            "Final answer: <short answer>\n"
            "Evidence: <quote or paraphrase from docs>\n"
            "If missing evidence, output:\n"
            "Final answer: Unknown from provided context.\n"
            "Evidence: Not found.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Documents:\n{context_block}\n\n"
            "Answer:"
        )
    else:
        prompt = (
            "You must answer using only the retrieved documents.\n"
            "Rules:\n"
            "1) First line must be: Final answer: <short answer, <= 40 words>\n"
            "2) If evidence is insufficient, write: Final answer: Unknown from provided context.\n"
            "3) Do not include any other lines.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved Documents:\n{context_block}\n\n"
            "Answer:"
        )
    raw = provider.generate([{"role": "user", "content": prompt}]).strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    if first_line.lower().startswith("final answer:"):
        return first_line[len("Final answer:"):].strip()
    return raw


def judge_openai(
    provider: OpenAIChatProvider,
    question: str,
    answer: str,
    contexts: Sequence[str],
    reference: str,
) -> Optional[Dict[str, float]]:
    context_block = "\n\n".join(f"[Doc {i+1}] {c}" for i, c in enumerate(contexts))
    prompt = (
        "You are evaluating a RAG answer. Return strict JSON with keys: "
        "faithfulness, relevance, correctness. Each value must be a float in [0,1].\n\n"
        f"Question: {question}\n\n"
        f"Retrieved Documents:\n{context_block}\n\n"
        f"Candidate Answer: {answer}\n\n"
        f"Reference Answer: {reference}\n"
    )
    raw = provider.generate([{"role": "user", "content": prompt}]).strip()
    try:
        obj = json.loads(raw)
    except Exception:
        # Try extracting a JSON object from wrapped text/code fences.
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return {
        "faithfulness": float(obj.get("faithfulness", 0.0)),
        "relevance": float(obj.get("relevance", 0.0)),
        "correctness": float(obj.get("correctness", 0.0)),
    }


def run_sample(
    row: Dict,
    *,
    hg: Hologram,
    top_k: int,
    use_routed: bool,
    use_drift: bool,
    use_gravity: bool,
    glyph_layout: str,
    reranker: Optional[PairReranker],
    rerank_top_n: int,
    generator: str,
    prompt_style: str,
    gen_provider: Optional[OpenAIChatProvider],
    judge_provider: Optional[OpenAIChatProvider],
) -> Tuple[SampleResult, Optional[Dict[str, float]]]:
    reset_hologram(hg, use_gravity=use_gravity)

    documents: List[str] = list(row.get("documents", []))
    if glyph_layout == "document":
        for i, doc in enumerate(documents, start=1):
            glyph = f"ragbench:{row['id']}:doc:{i}"
            hg.glyphs.create(glyph, title=f"RAGBench {row['id']} Doc {i}")
            hg.add_text(
                glyph_id=glyph,
                text=doc,
                trace_id=f"doc:{i}",
                do_extract_concepts=False,
                add_to_field=use_gravity,
                skip_nlp=True,
                origin="benchmark",
            )
    else:
        glyph = f"ragbench:{row['id']}"
        hg.glyphs.create(glyph, title=f"RAGBench {row['id']}")
        for i, doc in enumerate(documents, start=1):
            hg.add_text(
                glyph_id=glyph,
                text=doc,
                trace_id=f"doc:{i}",
                do_extract_concepts=False,
                add_to_field=use_gravity,
                skip_nlp=True,
                origin="benchmark",
            )

    question = str(row.get("question", ""))
    reference = str(row.get("response", ""))

    t0 = time.time()
    if use_routed:
        hits = hg.search_routed(question, top_k=top_k)
    elif use_drift:
        packet = hg.search_with_drift(question, top_k_traces=top_k)
        hits = [(r["trace"], float(r["score"])) for r in packet["results"]]
    else:
        hits = hg.search_text(question, top_k=top_k)
    latency_ms = (time.time() - t0) * 1000

    if reranker is not None and hits:
        limited = hits[:max(rerank_top_n, 1)]
        reranked = reranker.rerank(question, [tr.content for tr, _ in limited])
        text_to_trace = {tr.content: tr for tr, _ in limited}
        hits = [
            (text_to_trace[text], score)
            for score, text in reranked
            if text in text_to_trace
        ]

    contexts: List[str] = []
    retrieved_doc_ids: List[int] = []
    for tr, _score in hits:
        contexts.append(tr.content)
        m = re.match(r"doc:(\d+)", str(tr.trace_id))
        if m:
            retrieved_doc_ids.append(int(m.group(1)))

    if generator == "openai":
        if gen_provider is None:
            raise RuntimeError("OPENAI_API_KEY is required for --generator openai")
        answer = generate_openai(gen_provider, question, contexts, prompt_style=prompt_style)
    else:
        answer = generate_extractive(question, contexts)

    all_relevant_sentence_keys = list(row.get("all_relevant_sentence_keys", []))

    sample = SampleResult(
        sample_id=str(row.get("id", "")),
        question=question,
        answer=answer,
        reference=reference,
        token_f1=token_f1(answer, reference),
        exact_match=exact_match(answer, reference),
        groundedness_proxy=groundedness_proxy(answer, contexts),
        question_coverage=question_coverage(answer, question),
        retrieval_doc_recall=retrieval_doc_recall(retrieved_doc_ids, all_relevant_sentence_keys),
        latency_ms=latency_ms,
    )

    judge_scores = None
    if judge_provider is not None:
        judge_scores = judge_openai(judge_provider, question, answer, contexts, reference)

    return sample, judge_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end RAGBench benchmark for Hologram")
    parser.add_argument("--subset", default="hotpotqa", help="RAGBench subset, e.g. hotpotqa, msmarco")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-gravity", action="store_true", help="Enable gravity during indexing")
    parser.add_argument("--use-routed", action="store_true", help="Use glyph-routed retrieval")
    parser.add_argument("--use-drift", action="store_true", help="Use probe-based retrieval")
    parser.add_argument(
        "--glyph-layout",
        choices=["sample", "document"],
        default="sample",
        help="How documents are assigned to glyphs before retrieval",
    )
    parser.add_argument("--generator", choices=["extractive", "openai"], default="extractive")
    parser.add_argument("--gen-model", default="gpt-4o-mini")
    parser.add_argument("--prompt-style", choices=["strict", "concise_evidence"], default="strict")
    parser.add_argument("--judge", choices=["none", "openai"], default="none")
    parser.add_argument("--judge-model", default="gpt-4o-mini")
    parser.add_argument("--reranker", choices=["none", "cross_encoder"], default="none")
    parser.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--rerank-top-n", type=int, default=20)
    parser.add_argument("--hf-home", default="/tmp/hf_home", help="Writable Hugging Face cache root")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.use_routed and args.use_drift:
        raise SystemExit("--use-routed and --use-drift are mutually exclusive")

    if load_dotenv is not None:
        load_dotenv()

    os.environ.setdefault("HF_HOME", args.hf_home)
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")

    print(f"Loading RAGBench subset={args.subset} split={args.split}")
    ds = load_dataset("rungalileo/ragbench", args.subset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))

    api_key = os.getenv("OPENAI_API_KEY")
    gen_provider = None
    if args.generator == "openai":
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --generator openai")
        gen_provider = OpenAIChatProvider(api_key=api_key, model=args.gen_model)

    judge_provider = None
    if args.judge == "openai":
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is required for --judge openai")
        judge_provider = OpenAIChatProvider(api_key=api_key, model=args.judge_model)

    reranker = None
    if args.reranker == "cross_encoder":
        reranker = PairReranker(args.reranker_model)

    hg = Hologram.init(
        encoder_mode="minilm",
        use_clip=False,
        use_gravity=args.use_gravity,
        auto_ingest_system=False,
    )

    sample_results: List[SampleResult] = []
    judge_rows: List[Dict[str, float]] = []

    start_all = time.time()
    for idx, row in enumerate(ds, start=1):
        sres, jres = run_sample(
            row,
            hg=hg,
            top_k=args.top_k,
            use_routed=args.use_routed,
            use_drift=args.use_drift,
            use_gravity=args.use_gravity,
            glyph_layout=args.glyph_layout,
            reranker=reranker,
            rerank_top_n=args.rerank_top_n,
            generator=args.generator,
            prompt_style=args.prompt_style,
            gen_provider=gen_provider,
            judge_provider=judge_provider,
        )
        sample_results.append(sres)
        if jres:
            judge_rows.append(jres)

        if idx % 5 == 0:
            print(f"Processed {idx}/{len(ds)}")

    elapsed = time.time() - start_all

    def mean_attr(name: str) -> float:
        vals = [getattr(r, name) for r in sample_results]
        return statistics.mean(vals) if vals else 0.0

    # Dataset-provided reference quality (for context only)
    ref_adherence = [1.0 if bool(row.get("adherence_score", False)) else 0.0 for row in ds]
    ref_relevance = [float(row.get("relevance_score", 0.0) or 0.0) for row in ds]
    ref_completeness = [float(row.get("completeness_score", 0.0) or 0.0) for row in ds]
    ref_ragas_faith = [float(row.get("ragas_faithfulness", 0.0) or 0.0) for row in ds]
    ref_ragas_ctx_rel = [float(row.get("ragas_context_relevance", 0.0) or 0.0) for row in ds]

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "RAGBench-E2E",
        "dataset": "rungalileo/ragbench",
        "subset": args.subset,
        "split": args.split,
        "limit": len(sample_results),
        "config": {
            "top_k": args.top_k,
            "use_gravity": args.use_gravity,
            "use_routed": args.use_routed,
            "use_drift": args.use_drift,
            "glyph_layout": args.glyph_layout,
            "generator": args.generator,
            "gen_model": args.gen_model if args.generator == "openai" else None,
            "prompt_style": args.prompt_style if args.generator == "openai" else None,
            "judge": args.judge,
            "judge_model": args.judge_model if args.judge == "openai" else None,
            "reranker": args.reranker,
            "reranker_model": args.reranker_model if args.reranker != "none" else None,
            "rerank_top_n": args.rerank_top_n,
        },
        "timing": {
            "total_seconds": elapsed,
            "seconds_per_sample": _safe_div(elapsed, len(sample_results)),
            "retrieval_latency_ms": mean_attr("latency_ms"),
        },
        "metrics": {
            "token_f1_vs_reference": mean_attr("token_f1"),
            "exact_match_vs_reference": mean_attr("exact_match"),
            "groundedness_proxy": mean_attr("groundedness_proxy"),
            "question_coverage": mean_attr("question_coverage"),
            "retrieval_doc_recall": mean_attr("retrieval_doc_recall"),
        },
        "reference_response_metrics_on_same_slice": {
            "adherence_rate": statistics.mean(ref_adherence) if ref_adherence else 0.0,
            "relevance_score": statistics.mean(ref_relevance) if ref_relevance else 0.0,
            "completeness_score": statistics.mean(ref_completeness) if ref_completeness else 0.0,
            "ragas_faithfulness": statistics.mean(ref_ragas_faith) if ref_ragas_faith else 0.0,
            "ragas_context_relevance": statistics.mean(ref_ragas_ctx_rel) if ref_ragas_ctx_rel else 0.0,
        },
        "samples": [
            {
                "id": r.sample_id,
                "question": r.question,
                "answer": r.answer,
                "reference": r.reference,
                "token_f1": r.token_f1,
                "exact_match": r.exact_match,
                "groundedness_proxy": r.groundedness_proxy,
                "question_coverage": r.question_coverage,
                "retrieval_doc_recall": r.retrieval_doc_recall,
                "latency_ms": r.latency_ms,
            }
            for r in sample_results
        ],
    }

    if judge_rows:
        summary["judge_metrics"] = {
            "faithfulness": statistics.mean([x["faithfulness"] for x in judge_rows]),
            "relevance": statistics.mean([x["relevance"] for x in judge_rows]),
            "correctness": statistics.mean([x["correctness"] for x in judge_rows]),
        }

    out = Path(args.output) if args.output else Path(
        f"perf/ragbench_e2e_{args.subset}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nBenchmark complete")
    print(f"Output: {out}")
    print(f"Token-F1(ref):         {summary['metrics']['token_f1_vs_reference']:.4f}")
    print(f"Groundedness(proxy):   {summary['metrics']['groundedness_proxy']:.4f}")
    print(f"Retrieval Doc Recall:  {summary['metrics']['retrieval_doc_recall']:.4f}")


if __name__ == "__main__":
    main()
