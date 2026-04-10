# Benchmarking Guide

Last updated: 2026-04-11

This repo now has several benchmark paths. They do not answer the same question, so keep them separate.

## Benchmark Types

### Routed retrieval controls

Use these first when evaluating glyph routing itself:

- `tests/benchmark-glyph-routing-vs-global.py`
- `tests/benchmark-glyph-routing-minilm-real-text.py`

Use the synthetic benchmark to validate routing mechanics and shard behavior.

Use the MiniLM real-text benchmark to decide whether a routing or basis change actually improves retrieval quality on realistic embeddings.

### External retrieval benchmarks

Use these when you want public-dataset retrieval numbers:

- `scripts/benchmark_lotte_multidomain.py`
- `scripts/benchmark_beir.py`
- `scripts/benchmark_timeqa_e2e.py`
- `scripts/benchmark_timeqa_stream.py`

### End-to-end QA benchmarks

Use these when retrieval is only one part of the system being tested:

- `scripts/benchmark_ragbench_e2e.py`

## Recommended Evaluation Order

1. Validate mechanics with the synthetic routed benchmark
2. Check quality on the MiniLM real-text benchmark
3. For routing/compression architecture decisions, compare on mixed-domain LoTTE
4. If the idea still looks good, compare on public datasets such as BEIR
5. Only then move to end-to-end QA benchmarks

This avoids treating generator noise as a retrieval result.

## Latency Interpretation

Be careful with routed-query timing.

Routed retrieval can include:
- lazy shard construction
- operator initialization
- projection setup
- trace transformation and FAISS index build

Because of that, the first routed query is often a cold-path measurement, not a steady-state query measurement.

For projected operators, cold-path timing can include substantial setup work that should not be compared directly to warm query latency.

For PCA benchmarks, be explicit about whether timing includes:
- PCA fit/build time
- query-time search only

The current LoTTE PCA numbers in this repo are query-time only after the compressed index is built.

## Metrics That Matter Most

For routed retrieval experiments, prioritize:
- recall
- interference rate
- routing accuracy
- warm query latency

For public retrieval benchmarks, prioritize:
- `NDCG@10`
- `Recall@100`
- `MRR@10`

For mixed-domain LoTTE architecture benchmarks, prioritize:
- `NDCG@10`
- `Recall@10`
- `MAP@10`
- warm `ms/query`
- whether the comparison isolates:
  - routing loss
  - merge loss
  - compression loss

For end-to-end QA benchmarks, separate:
- retrieval quality
- answer quality
- grounding / support metrics

## Minimal Command Set

```bash
python tests/benchmark-glyph-routing-vs-global.py
python tests/benchmark-glyph-routing-minilm-real-text.py
python scripts/benchmark_lotte_multidomain.py --mode global
python scripts/benchmark_lotte_multidomain.py --mode routed --top-glyphs 2
python scripts/benchmark_lotte_multidomain.py --mode global_pca --pca-dim 64
python scripts/benchmark_beir.py --dataset scifact --split test --download
python scripts/benchmark_ragbench_e2e.py --subset hotpotqa --split test --limit 30 --top-k 5 --generator extractive
```

## Current LoTTE Snapshot

On the current mixed-domain `LoTTE` benchmark slice used in this repo:

- quality ceiling:
  - global `384D`: `0.926` `NDCG@10`, `~98 ms/query`
- best fast path at current scale:
  - global `PCA-64`: `0.907` `NDCG@10`, `~17 ms/query`
- best refined routed path:
  - same-space routed + weighted/filter merge: about `0.894` `NDCG@10`, `~17-18 ms/query`

Important negative results already established:
- operator-projected routed retrieval regresses quality badly
- whitening does not beat same-space routed
- confidence gating does not pay off
- naive hybrid lexical fusion regresses quality
- the tested cross-encoder reranker is much slower and did not improve quality

Important positive results already established:
- same-space routed retrieval fixed the main routed bug
- shard-weighted merge helps a bit
- post-retrieval shard2 filtering helps a bit
- moderate global compression (`PCA-64`) is a legitimate speed lever

## Reporting Guidance

When writing up results:
- say whether the benchmark is synthetic, real-text, public retrieval, or end-to-end QA
- say whether timing is cold or warm
- say whether timing includes index/projection build or query-time only
- do not describe a shared discriminant basis as full per-glyph learned `R_g`
- do not treat GLiNER or symbolic extraction changes as retrieval wins unless the benchmark actually tests them
- do not assume effective dimensionality for variance implies the same dimensionality for retrieval discrimination
