# BEIR Benchmark for External RAG Comparison

This project now includes `scripts/benchmark_beir.py`, which runs Hologram against a standard BEIR retrieval dataset.

Why BEIR:
- It is a widely used public benchmark suite for zero-shot retrieval.
- You can compare your `NDCG@10` and `Recall@100` to published baselines and leaderboard submissions.

## Install

```bash
./.venv/bin/pip install beir
```

## Run (real benchmark)

```bash
./.venv/bin/python scripts/benchmark_beir.py \
  --dataset scifact \
  --split test \
  --download \
  --encoder-mode minilm \
  --top-k 100
```

Output is written to `perf/beir_<dataset>_<timestamp>.json`.

## Compare retrieval modes

Standard similarity search:

```bash
./.venv/bin/python scripts/benchmark_beir.py \
  --dataset scifact --split test --download \
  --encoder-mode minilm --top-k 100
```

Probe-based retrieval:

```bash
./.venv/bin/python scripts/benchmark_beir.py \
  --dataset scifact --split test --download \
  --encoder-mode minilm --use-gravity --use-drift --top-k 100
```

## Smoke test before a full run

```bash
./.venv/bin/python scripts/benchmark_beir.py \
  --dataset scifact --split test --download \
  --max-docs 2000 --max-queries 50 --top-k 100
```

## Main metrics to track

- `NDCG@10`: primary BEIR leaderboard metric
- `Recall@100`: retrieval coverage for downstream generation
- `MRR@10`: early precision
- latency: `timing.query_ms`

## Public references for comparison

- BEIR paper: https://arxiv.org/abs/2104.08663
- BEIR code and datasets: https://github.com/beir-cellar/beir
- BEIR leaderboard entry point: https://eval.ai/web/challenges/challenge-page/1033/overview

## Notes for fair comparisons

- Use the same dataset and split (`test`) as leaderboard reports.
- Keep `top-k` fixed across runs when comparing system variants.
- If you evaluate a full RAG pipeline later, keep retrieval fixed and vary only generation/reranking in separate experiments.
