#!/usr/bin/env python3
"""Benchmark mixed-domain retrieval on a union of BEIR datasets.

This is a more routing-relevant benchmark than single-dataset BEIR:
- each dataset is ingested as its own glyph/domain
- queries run against the full mixed corpus
- global retrieval competes across all domains
- routed retrieval can route toward the correct dataset glyph
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


def load_beir_dataset(data_dir: Path, dataset: str, split: str, download: bool):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

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


def ingest_dataset(
    hg: Hologram,
    dataset: str,
    corpus: Dict[str, Dict[str, str]],
    max_docs: int | None,
) -> int:
    glyph_id = f"beir:{dataset}"
    hg.glyphs.create(glyph_id, title=f"BEIR {dataset}")

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
            trace_id=f"{dataset}:doc:{doc_id}",
            do_extract_concepts=False,
            add_to_field=False,
            skip_nlp=True,
            origin="benchmark",
        )
        count += 1
    return count


def run_queries(
    hg: Hologram,
    all_queries: List[Tuple[str, str]],
    top_k: int,
    mode: str,
) -> Results:
    results: Results = {}
    for qid, query_text in all_queries:
        if mode == "routed":
            hits = hg.search_routed(query_text, top_k=top_k)
        elif mode == "adaptive":
            hits = hg.search_adaptive(query_text, top_k=top_k)
        else:
            hits = hg.search_text(query_text, top_k=top_k)
        results[qid] = {str(trace.trace_id): float(score) for trace, score in hits}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark routed retrieval on a mixed-domain BEIR corpus")
    parser.add_argument(
        "--datasets",
        default="scifact,trec-covid",
        help="Comma-separated BEIR datasets to union into one corpus",
    )
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--data-dir", default="data/beir")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--encoder-mode", default="minilm", choices=["minilm", "hash", "clip", "default"])
    parser.add_argument("--mode", choices=["global", "routed", "adaptive"], default="global")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-docs-per-dataset", type=int, default=1000)
    parser.add_argument("--max-queries-per-dataset", type=int, default=25)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if len(datasets) < 2:
        raise SystemExit("Use at least two datasets so routing has real cross-domain competition.")

    hg = Hologram.init(
        encoder_mode=args.encoder_mode,
        use_clip=(args.encoder_mode in {"clip", "default"}),
        use_gravity=False,
        auto_ingest_system=False,
    )

    data_dir = Path(args.data_dir)
    all_queries: List[Tuple[str, str]] = []
    qrels: Qrels = {}
    dataset_sizes: Dict[str, Dict[str, int]] = {}

    ingest_start = time.time()
    for dataset in datasets:
        corpus, queries, raw_qrels = load_beir_dataset(
            data_dir=data_dir,
            dataset=dataset,
            split=args.split,
            download=args.download,
        )
        indexed_docs = ingest_dataset(hg, dataset, corpus, args.max_docs_per_dataset)
        indexed_doc_ids = set(list(corpus.keys())[:indexed_docs])

        kept_queries = list(queries.items())[: args.max_queries_per_dataset]
        for query_id, query_text in kept_queries:
            prefixed_qid = f"{dataset}:q:{query_id}"
            filtered_rels = {
                f"{dataset}:doc:{doc_id}": int(rel)
                for doc_id, rel in raw_qrels.get(query_id, {}).items()
                if doc_id in indexed_doc_ids
            }
            if not filtered_rels:
                continue
            qrels[prefixed_qid] = filtered_rels
            all_queries.append((prefixed_qid, query_text))

        dataset_sizes[dataset] = {
            "indexed_docs": indexed_docs,
            "queries_kept": sum(1 for qid, _ in kept_queries if f"{dataset}:q:{qid}" in qrels),
        }

    ingest_s = time.time() - ingest_start

    search_start = time.time()
    results = run_queries(hg, all_queries, top_k=args.top_k, mode=args.mode)
    search_s = time.time() - search_start

    metrics = compute_metrics(qrels=qrels, results=results, ks=[1, 3, 5, 10])
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": "BEIR-MultiDomain",
        "datasets": datasets,
        "split": args.split,
        "mode": args.mode,
        "config": {
            "top_k": args.top_k,
            "max_docs_per_dataset": args.max_docs_per_dataset,
            "max_queries_per_dataset": args.max_queries_per_dataset,
            "encoder_mode": args.encoder_mode,
        },
        "dataset_sizes": dataset_sizes,
        "timing": {
            "ingest_seconds": ingest_s,
            "query_seconds": search_s,
            "query_ms_per_query": (search_s / max(len(all_queries), 1)) * 1000.0,
        },
        "metrics": metrics,
        "queries_evaluated": len(all_queries),
    }

    output = Path(args.output) if args.output else Path(
        f"perf/beir_multidomain_{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))

    print("Benchmark complete")
    print(f"Output: {output}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Queries evaluated: {len(all_queries)}")
    print(f"NDCG@10: {payload['metrics']['ndcg'].get('NDCG@10', 0.0):.4f}")
    print(f"Recall@10: {payload['metrics']['recall'].get('Recall@10', 0.0):.4f}")
    print(f"MRR@10: {payload['metrics']['mrr'].get('MRR@10', 0.0):.4f}")
    print(f"Query ms/query: {payload['timing']['query_ms_per_query']:.2f}")


if __name__ == "__main__":
    main()
