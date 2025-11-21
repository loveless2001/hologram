# Benchmarking Your Holographic Memory System

## Quick Start

```bash
python benchmark.py
```

This will run the full benchmark suite and output results to `benchmark_results.json`.

---

## What Gets Measured?

### 1. **Retrieval Quality** (How good is search?)
- **Precision@10**: Of the top 10 results, how many are relevant?
- **Recall@10**: Of all relevant items, how many are in top 10?
- **MRR (Mean Reciprocal Rank)**: How quickly is the first relevant result found?

**Target**: Precision > 0.7, Recall > 0.5, MRR > 0.6

### 2. **Gravity Field Behavior** (Does physics work?)
- **Attraction Score**: Do positive statements pull concepts together?
- **Repulsion Score**: Do negative statements push concepts apart?

**Target**: Attraction > 1.0, Repulsion < 1.0

### 3. **Temporal Dynamics** (Does memory decay work?)
- **Decay Half-Life**: How many steps until concept loses 50% mass?
- **Reinforcement Gain**: How much do reinforced concepts strengthen?

**Target**: Half-life 20-50 steps, Reinforcement gain 1.5-2.0x

### 4. **Efficiency** (How fast?)
- **Query Latency**: Average time to search (milliseconds)
- **Indexing Throughput**: Concepts added per second

**Target**: Latency < 100ms, Throughput > 100 concepts/sec

---

## Custom Benchmarks

### Example: Test Your Own Knowledge Base

```python
from hologram import Hologram
from benchmark import HologramBenchmark

# Build your knowledge base
hg = Hologram.init(use_clip=False)
hg.glyphs.create("physics", title="Physics")

# Add knowledge
hg.add_text("physics", "Mass increases with velocity")
hg.add_text("physics", "Energy equals mass times c squared")
# ... add more

# Define test queries with ground truth
test_data = {
    'retrieval_queries': [
        ("what happens to mass at high speed", ["trace_id_1", "trace_id_2"]),
        ("energy mass relationship", ["trace_id_3"]),
    ],
    'positive_triplets': [
        ("Mass", "increases with", "Velocity"),
    ],
    'negative_triplets': [
        ("Mass", "independent of", "Color"),
    ],
    'performance_queries': [
        "what is energy",
        "how does mass behave",
    ]
}

# Run benchmark
benchmark = HologramBenchmark(hg)
results = benchmark.run_full_suite(test_data)
```

---

## Interpreting Results

### Good Signs ✓
- Attraction score > 1.0: Concepts drift together when linked
- Repulsion score < 1.0: Negation pushes concepts apart
- Reinforcement gain > 1.5: Frequent access keeps concepts strong
- Query latency < 50ms: Fast enough for interactive use

### Warning Signs ⚠️
- Attraction score < 1.0: Gravity field may not be working
- Decay half-life = None: Concepts not decaying (check isolation_drift param)
- Very low precision/recall: May need better embeddings or more data

### Red Flags ✗
- Both attraction AND repulsion > 1.0: Something is broken
- Query latency > 500ms: FAISS index may need optimization
- Reinforcement gain < 1.0: Concepts losing mass despite reinforcement

---

## Comparing Against Baselines

To compare with traditional approaches:

```python
# Your holographic system
hg = Hologram.init(use_clip=False)
# ... add data ...
bench_holo = HologramBenchmark(hg)
results_holo = bench_holo.run_full_suite(test_data)

# Pure vector DB (no gravity, no decay)
hg_baseline = Hologram.init(use_clip=False, use_gravity=False)
# ... add same data ...
bench_baseline = HologramBenchmark(hg_baseline)
results_baseline = bench_baseline.run_full_suite(test_data)

# Compare
print(f"Holographic MRR: {results_holo['MRR']:.3f}")
print(f"Baseline MRR:    {results_baseline['MRR']:.3f}")
```

---

## Standard Test Datasets

### 1. **Synthetic Cluster Test**

Tests if concepts naturally cluster by topic:

```python
# Create 3 clusters
physics_concepts = ["gravity", "mass", "velocity", ...]
biology_concepts = ["cell", "DNA", "protein", ...]
history_concepts = ["war", "treaty", "empire", ...]

# Add all
for c in physics_concepts + biology_concepts + history_concepts:
    hg.add_text("test", c)

# Measure: Do physics concepts cluster together?
# Use silhouette score or manual inspection
```

### 2. **Negation Test**

Tests if negation creates repulsion:

```python
test_data = {
    'negative_triplets': [
        ("Hot", "opposite of", "Cold"),
        ("Up", "opposite of", "Down"),
        ("Good", "opposite of", "Bad"),
    ]
}

# Expected: Repulsion score < 1.0 (concepts push apart)
```

### 3. **Temporal Decay Test**

Tests if old concepts fade:

```python
# Add concept, then add 100 other concepts without reinforcing it
# Measure: mass should decrease over time
```

---

## Metrics Dashboard

For continuous monitoring, track these over time:

| Metric | Target | Critical | Notes |
|--------|--------|----------|-------|
| Precision@10 | > 0.7 | < 0.3 | Core retrieval quality |
| MRR | > 0.6 | < 0.2 | First result quality |
| Attraction | > 1.0 | < 0.9 | Gravity working? |
| Repulsion | < 1.0 | > 1.1 | Negation working? |
| Query latency | < 100ms | > 500ms | User experience |
| Half-life | 20-50 | < 5 or > 200 | Decay tuning |

---

## Contributing Your Results

If you run benchmarks on interesting datasets, please share:

1. Run benchmark: `python benchmark.py`
2. Export results: `benchmark_results.json`
3. Document:
   - Dataset size
   - Domain (physics, history, etc.)
   - Any custom parameters
4. Submit as GitHub issue or PR

Example format:
```markdown
## Benchmark Results: Wikipedia Physics (1000 concepts)

- Precision@10: 0.82
- Recall@10: 0.65
- MRR: 0.74
- Attraction: 1.15
- Repulsion: 0.87
- Avg latency: 12ms

Notes: Used CLIP embeddings, decay params at default.
```

---

## Troubleshooting

**Q: Attraction score is < 1.0, why?**  
A: Check if `use_gravity=True` in `Hologram.init()`. Also verify `eta` parameter.

**Q: Nothing is decaying (half-life = None)?**  
A: Need to call `hg.decay(steps=N)` or increase `isolation_drift` parameter.

**Q: Query latency very high?**  
A: FAISS should be sub-linear. Check dataset size and indexing method.

**Q: Low precision/recall?**  
A: May need:
- Better embeddings (try use_clip=True with actual CLIP)
- More data (< 100 concepts may not cluster well)
- Tuned decay parameters

---

For detailed metric definitions, see `/docs/performance_metrics.md`.
