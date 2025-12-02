# MG Scorer Module

## Overview

The **Memory Gravity (MG) Scorer** is a geometric analysis module that quantifies the health and stability of semantic vector fields. Introduced in Hologram v1.2.0, it provides real-time quality metrics for memory clusters, LLM outputs, and retrieval results.

### Core Metrics

- **Coherence**: How tightly clustered vectors are (0-1, higher is better)
- **Curvature**: How linear/smooth the semantic trajectory is (0-1, 1.0 = perfectly linear)
- **Entropy**: Semantic dispersion across principal components (lower = more focused)
- **Collapse Risk**: Composite instability indicator (lower = more stable)
- **Gradient**: Direction of semantic tension in vector space

## Installation

The MG Scorer is included in the `hologram` package. No additional dependencies are required beyond NumPy.

## Quick Start

```python
from hologram import Hologram

# Initialize Hologram with MiniLM encoder (recommended for semantic quality)
hg = Hologram.init(encoder_mode="minilm", use_gravity=False)

# Score a list of texts
texts = [
    "The cat is sleeping on the couch.",
    "The cat is resting on the sofa.",
    "The cat lies quietly on the couch."
]

score = hg.score_text(texts)

print(f"Coherence: {score.coherence:.4f}")  # How similar are these texts?
print(f"Entropy:   {score.entropy:.4f}")    # How dispersed is the meaning?
print(f"Curvature: {score.curvature:.4f}")  # How linear is the semantic flow?
print(f"Risk:      {score.collapse_risk:.4f}")  # How unstable is this cluster?
```

## Metric Definitions

### 1. Coherence
**Definition**: `1 - average_cosine_distance(all_pairs)`

**Range**: [0, 1]

**Interpretation**:
- **> 0.8**: Highly coherent (tight cluster, paraphrases)
- **0.5 - 0.8**: Moderately coherent (related topics)
- **< 0.5**: Low coherence (random or unrelated)

**Example**:
```python
# High coherence (0.73)
texts = [
    "The cat is sleeping on the couch.",
    "The cat is resting on the sofa.",
    "The cat lies quietly on the couch."
]

# Low coherence (0.004)
texts = [
    "Bananas contain potassium.",
    "Quantum entanglement challenges classical locality.",
    "My shoes got wet yesterday."
]
```

### 2. Curvature
**Definition**: Straightness index using triangle inequality on ordered sequences.

**Formula**: For triplet (v₀, v₁, v₂):
```
κ = ||v₂ - v₀|| / (||v₁ - v₀|| + ||v₂ - v₁||)
```

**Range**: [0, 1]

**Interpretation**:
- **1.0**: Perfectly linear semantic progression
- **0.7**: 90° semantic turn
- **< 0.6**: Sharp topic shift or contradiction

**Example**:
```python
# Linear argument (curvature 0.71)
texts = [
    "Climate change leads to more extreme weather events.",
    "Extreme weather events increase infrastructure damage.",
    "Infrastructure damage raises national recovery costs."
]

# Sharp turn (curvature 0.55)
texts = [
    "Machine learning models improve by analyzing patterns.",
    "Deep neural networks extract nonlinear features.",
    "Medieval architecture used flying buttresses."  # Topic shift!
]
```

### 3. Semantic Entropy
**Definition**: Eigenvalue-based dispersion measure using covariance matrix.

**Formula**:
```
H = -Σ λᵢ log(λᵢ)
```
where λᵢ are normalized eigenvalues of cov(vectors).

**Range**: [0, ∞)

**Interpretation**:
- **< 0.5**: Tight semantic cluster (single topic)
- **0.5 - 1.5**: Moderate variation (multiple related aspects)
- **> 1.5**: High dispersion (many different directions)

### 4. Collapse Risk
**Definition**: Composite instability indicator.

**Formula**:
```
risk = (1 - coherence) × log(1 + entropy) × (1 - curvature)
```

**Range**: [0, ∞)

**Interpretation**:
- **< 0.1**: Stable, coherent configuration
- **0.1 - 0.3**: Moderate risk (some confusion)
- **> 0.3**: High risk (contradictory or fragmenting)

### 5. Gradient
**Definition**: Average deviation from centroid.

**Formula**:
```
∇ = mean(vᵢ - centroid)
```

**Range**: ℝᴰ (vector in embedding space)

**Interpretation**: Direction of semantic tension. Points toward the "pull" of the field.

## API Reference

### `Hologram.score_text(texts: List[str]) -> MGScore`

Compute MG scores for a list of text strings.

**Parameters:**
- `texts`: List of strings to analyze

**Returns:**
- `MGScore` dataclass with all metrics

**Use cases:**
- Evaluate coherence of generated paragraphs
- Detect semantic drift in conversations
- Measure quality of document summaries

### `Hologram.score_trace(trace_ids: List[str]) -> MGScore`

Compute MG scores for a list of trace IDs from the memory store.

**Parameters:**
- `trace_ids`: List of trace IDs to analyze

**Returns:**
- `MGScore` dataclass with all metrics

**Use cases:**
- Evaluate coherence of retrieval results
- Assess quality of memory clusters
- Monitor field stability over time

### `MGScore` Dataclass

```python
@dataclass
class MGScore:
    coherence: float       # [0, 1], higher = more coherent
    curvature: float       # [0, 1], higher = more linear
    entropy: float         # [0, ∞), lower = more focused
    collapse_risk: float   # [0, ∞), lower = more stable
    gradient: np.ndarray   # Direction of semantic tension
```

### Direct Scoring

For advanced use, you can score raw vectors:

```python
from hologram.mg_scorer import mg_score
import numpy as np

vectors = np.random.rand(10, 128)
score = mg_score(vectors)
```

## Use Cases

### 1. LLM Output Validation

```python
# Generated paragraph from LLM
generated = [
    "First sentence.",
    "Second sentence.",
    "Third sentence."
]

score = hg.score_text(generated)

if score.coherence < 0.6:
    print("Warning: Generated text is incoherent!")
if score.collapse_risk > 0.2:
    print("Warning: Text may be hallucinating or contradicting itself!")
```

### 2. Memory Field Health Monitoring

```python
# Get recent traces
recent_traces = hg.store.get_recent_traces(n=50)
trace_ids = [t.trace_id for t in recent_traces]

score = hg.score_trace(trace_ids)

if score.collapse_risk > 0.3:
    print("Field is fragmenting! Consider running decay or cleanup.")
    hg.step_decay(steps=10)  # Trigger cleanup
```

### 3. Topic Drift Detection

```python
# Analyze conversation flow
messages = [
    "Let's discuss the project deadline.",
    "We need to finish by Friday.",
    "Friday is also my birthday!",  # Drift starts
    "Birthdays are fun celebrations."
]

score = hg.score_text(messages)

if score.curvature < 0.6:
    print(f"Topic drift detected! Curvature: {score.curvature:.2f}")
```

### 4. Retrieval Quality Gate

```python
# Ensure retrieval results are coherent
query = "What causes climate change?"
results = hg.search_text(query, top_k=20)
result_ids = [r[0].trace_id for r in results]

score = hg.score_trace(result_ids)

if score.coherence < 0.5:
    print("Warning: Retrieved results are not coherent!")
    # Fall back to fewer results or re-rank
    results = results[:10]
```

## Performance Characteristics

- **Coherence**: O(N²) - All-pairs distance matrix
- **Curvature**: O(N) - Sequential triplet analysis
- **Entropy**: O(N × D²) - Eigenvalue decomposition
- **Overall**: Fast for N < 1000, D < 500

For large batches (N > 1000), consider:
- Sampling
- Batch processing
- Parallel scoring

## Testing

The module includes comprehensive test coverage:

- **Unit Tests**: `tests/test_mg_scorer.py`
- **Scenario Tests**: `tests/test_mg_scenarios.py`
- **Chaos Tests**: `tests/test_mg_chaos.py`

Run tests:
```bash
pytest tests/test_mg_*.py -v
```

## Limitations

- **Requires ordered sequences** for meaningful curvature analysis
- **Sensitive to outliers** in small samples (N < 5)
- **Encoder quality matters**: Hash encoders will show lower coherence than semantic encoders (MiniLM, CLIP)

## Future Extensions (v1.3+)

- Glyph-weighted coherence (emphasize important concepts)
- Multi-scale scoring (sentence → paragraph → document)
- Temporal decay effects (older vectors weighted less)
- Graph Laplacian curvature (for non-sequential sets)
- Real-time streaming scorer for live LLM monitoring

## References

- Design Document: `hologram/mgscore.md`
- Test Scenarios: `hologram/mgscore_testing.md`
- Implementation: `hologram/mg_scorer.py`
- Architecture: `docs/architecture_design.md`
- Research Paper: `docs/research_paper.md`
