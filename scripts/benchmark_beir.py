#!/usr/bin/env python3
"""Run Hologram against a BEIR retrieval benchmark dataset.

This script evaluates retrieval quality using standard IR metrics so results are
comparable to BEIR leaderboard baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from hologram.api import Hologram


Qrels = Dict[str, Dict[str, int]]
Results = Dict[str, Dict[str, float]]


def compute_metrics(qrels: Qrels, results: Results, ks: List[int]) -> Dict[str, Dict[str, float]]:
    try:
        from beir.retrieval.evaluation import EvaluateRetrieval
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: beir. Install with: ./.venv/bin/pip install beir"
        ) from exc

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


def load_beir_dataset(data_dir: Path, dataset: str, split: str, download: bool):
    try:
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Missing dependency: beir. Install with: ./.venv/bin/pip install beir"
        ) from exc

    dataset_dir = data_dir / dataset

    if download and not dataset_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
        archive_path = data_dir / f"{dataset}.zip"
        util.download_url(url, str(archive_path))
        util.unzip(str(archive_path), str(data_dir))

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset path not found: {dataset_dir}. Use --download or provide a prepared BEIR dataset."
        )

    corpus_raw, queries_raw, qrels_raw = GenericDataLoader(str(dataset_dir)).load(split=split)

    corpus = {str(doc_id): doc for doc_id, doc in corpus_raw.items()}
    queries = {str(query_id): text for query_id, text in queries_raw.items()}
    qrels: Qrels = {
        str(query_id): {str(doc_id): int(rel) for doc_id, rel in doc_rels.items()}
        for query_id, doc_rels in qrels_raw.items()
    }
    return corpus, queries, qrels


def ingest_corpus(
    hg: Hologram,
    glyph_id: str,
    corpus: Dict[str, Dict[str, str]],
    use_gravity: bool,
    max_docs: int | None,
) -> int:
    hg.glyphs.create(glyph_id, title=f"BEIR {glyph_id}")

    count = 0
    for doc_id, doc in corpus.items():
        if max_docs is not None and count >= max_docs:
            break

        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        content = f"{title}\n{text}".strip() if title else text

        hg.add_text(
            glyph_id=glyph_id,
            text=content,
            trace_id=str(doc_id),
            do_extract_concepts=False,
            add_to_field=use_gravity,
            skip_nlp=True,
            origin="benchmark",
        )
        count += 1

        if count % 1000 == 0:
            print(f"Indexed {count} documents...")

    return count


def run_retrieval(
    hg: Hologram,
    queries: Dict[str, str],
    top_k: int,
    use_drift: bool,
    max_queries: int | None,
) -> Tuple[Results, int]:
    results: Results = {}

    items = list(queries.items())
    if max_queries is not None:
        items = items[:max_queries]

    for idx, (qid, query_text) in enumerate(items, start=1):
        if use_drift:
            packet = hg.search_with_drift(query_text, top_k_traces=top_k)
            hits = [(row["trace"], row["score"]) for row in packet["results"]]
        else:
            hits = hg.search_text(query_text, top_k=top_k)

        results[str(qid)] = {str(trace.trace_id): float(score) for trace, score in hits}

        if idx % 100 == 0:
            print(f"Searched {idx} queries...")

    return results, len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Hologram on a BEIR dataset")
    parser.add_argument("--dataset", default="scifact", help="BEIR dataset name, e.g. scifact, fiqa, trec-covid")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"], help="BEIR split")
    parser.add_argument("--data-dir", default="data/beir", help="Directory where BEIR datasets are stored")
    parser.add_argument("--download", action="store_true", help="Download dataset if missing")
    parser.add_argument("--encoder-mode", default="minilm", choices=["minilm", "hash", "clip", "default"])
    parser.add_argument("--use-gravity", action="store_true", help="Enable gravity field while indexing")
    parser.add_argument("--use-drift", action="store_true", help="Use probe-based retrieval instead of standard search")
    parser.add_argument("--top-k", type=int, default=100, help="Number of docs to retrieve per query")
    parser.add_argument("--max-docs", type=int, default=None, help="Optional cap for quick smoke tests")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional cap for quick smoke tests")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print(f"Loading dataset: {args.dataset} ({args.split})")
    corpus, queries, qrels = load_beir_dataset(
        data_dir=data_dir,
        dataset=args.dataset,
        split=args.split,
        download=args.download,
    )

    print(f"Corpus size: {len(corpus)} | Queries: {len(queries)} | Qrels: {len(qrels)}")

    hg = Hologram.init(
        encoder_mode=args.encoder_mode,
        use_clip=(args.encoder_mode in {"clip", "default"}),
        use_gravity=args.use_gravity,
        auto_ingest_system=False,
    )

    ingest_start = time.time()
    indexed_docs = ingest_corpus(
        hg=hg,
        glyph_id=f"beir:{args.dataset}",
        corpus=corpus,
        use_gravity=args.use_gravity,
        max_docs=args.max_docs,
    )
    ingest_s = time.time() - ingest_start

    if args.max_docs is not None:
        indexed_doc_ids = set(list(corpus.keys())[: args.max_docs])
        qrels = {
            qid: {doc: rel for doc, rel in rels.items() if doc in indexed_doc_ids}
            for qid, rels in qrels.items()
        }

    search_start = time.time()
    results, searched_queries = run_retrieval(
        hg=hg,
        queries=queries,
        top_k=args.top_k,
        use_drift=args.use_drift,
        max_queries=args.max_queries,
    )
    search_s = time.time() - search_start

    if args.max_queries is not None:
        kept_query_ids = set(list(queries.keys())[: args.max_queries])
        qrels = {qid: rels for qid, rels in qrels.items() if qid in kept_query_ids}

    # Keep only queries that still have at least one relevant doc after filtering.
    qrels = {qid: rels for qid, rels in qrels.items() if rels}
    if not qrels:
        raise SystemExit(
            "No overlapping qrels after filtering. Increase --max-docs/--max-queries "
            "or run without caps."
        )
    # `evaluate_custom` expects every results key to exist in qrels.
    results = {qid: docs for qid, docs in results.items() if qid in qrels}
    if not results:
        raise SystemExit(
            "No overlapping results/qrels query IDs. Increase --max-queries "
            "or run without caps."
        )

    metrics = compute_metrics(
        qrels=qrels,
        results=results,
        ks=[1, 3, 5, 10, 100],
    )

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "BEIR",
        "dataset": args.dataset,
        "split": args.split,
        "encoder_mode": args.encoder_mode,
        "use_gravity": args.use_gravity,
        "use_drift": args.use_drift,
        "top_k": args.top_k,
        "indexed_docs": indexed_docs,
        "searched_queries": searched_queries,
        "timing": {
            "ingest_seconds": ingest_s,
            "search_seconds": search_s,
            "ingest_docs_per_sec": indexed_docs / ingest_s if ingest_s > 0 else 0.0,
            "query_ms": (search_s / searched_queries) * 1000 if searched_queries else 0.0,
        },
        "metrics": metrics,
    }

    output_path = Path(args.output) if args.output else Path(
        f"perf/beir_{args.dataset}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nBenchmark complete")
    print(f"Output: {output_path}")
    print(f"NDCG@10:   {payload['metrics']['ndcg']['NDCG@10']:.4f}")
    print(f"Recall@100:{payload['metrics']['recall']['Recall@100']:.4f}")
    print(f"MRR@10:    {payload['metrics']['mrr']['MRR@10']:.4f}")
    print(f"Avg query: {payload['timing']['query_ms']:.2f} ms")


if __name__ == "__main__":
    main()
