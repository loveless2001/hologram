# End-to-End RAGBench Benchmark

This project includes `scripts/benchmark_ragbench_e2e.py` for question-answering quality evaluation on a real public benchmark dataset:
- Dataset: `rungalileo/ragbench`
- Supports subsets like `hotpotqa`

## What it evaluates

Per sample:
1. Ingest benchmark documents into Hologram
2. Retrieve top-k context (`search_text` or `search_with_drift`)
3. Generate answer (`extractive` baseline or OpenAI model)
4. Score output

Runner enhancements:
- Reuses a single initialized Hologram instance across samples (faster runs)
- Optional reranking (`--reranker cross_encoder`)
- Strict OpenAI answer formatting (`Final answer: ...`)

Metrics reported:
- `token_f1_vs_reference`
- `exact_match_vs_reference`
- `groundedness_proxy` (answer tokens supported by retrieved context)
- `question_coverage` (question keywords covered in answer)
- `retrieval_doc_recall` (gold relevant docs retrieved)

Also reports dataset-provided reference metrics (for context only) on the same slice:
- `adherence_rate`, `relevance_score`, `completeness_score`
- `ragas_faithfulness`, `ragas_context_relevance`

## Quick run (no API key needed)

```bash
./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa \
  --split test \
  --limit 30 \
  --top-k 5 \
  --generator extractive
```

## OpenAI generation run

```bash
OPENAI_API_KEY=... ./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa \
  --split test \
  --limit 30 \
  --top-k 5 \
  --generator openai \
  --gen-model gpt-4o-mini
```

## Optional LLM judge mode

```bash
OPENAI_API_KEY=... ./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa \
  --split test \
  --limit 30 \
  --top-k 5 \
  --generator openai \
  --gen-model gpt-4o-mini \
  --judge openai \
  --judge-model gpt-4o-mini
```

## Compare retrieval modes

Standard retrieval:

```bash
./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa --split test --limit 30 --top-k 5
```

Probe-based retrieval:

```bash
./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa --split test --limit 30 --top-k 5 \
  --use-gravity --use-drift
```

With reranking:

```bash
./.venv/bin/python scripts/benchmark_ragbench_e2e.py \
  --subset hotpotqa --split test --limit 30 --top-k 20 \
  --reranker cross_encoder --rerank-top-n 20
```

Output is saved under `perf/ragbench_e2e_<subset>_<timestamp>.json` unless `--output` is provided.

Note: the script defaults `--hf-home /tmp/hf_home` so dataset cache/locks stay in a writable location.
