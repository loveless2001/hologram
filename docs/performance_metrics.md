# Performance Metrics for Holographic Memory

## Overview

This document defines metrics to evaluate the holographic memory system across multiple dimensions: retrieval quality, gravity field behavior, temporal dynamics, and comparison with baseline systems.

---

## 1. Retrieval Quality Metrics

### 1.1 Standard IR Metrics

**Precision@K**
```
Precision@K = (# relevant results in top K) / K
```
Measures: How many retrieved results are actually relevant?

**Recall@K**
```
Recall@K = (# relevant results in top K) / (total # relevant items)
```
Measures: How many of all relevant items were retrieved?

**Mean Reciprocal Rank (MRR)**
```
MRR = (1/|Q|) × Σ(1/rank_of_first_relevant_result)
```
Measures: How quickly does the system return the first relevant result?

**Normalized Discounted Cumulative Gain (NDCG@K)**
```
NDCG@K = DCG@K / IDCG@K
where DCG@K = Σ(relevance_i / log2(i+1))
```
Measures: Quality of ranking (rewards relevant items at top positions)

### 1.2 Domain-Specific Metrics

**Conceptual Coverage**
```
Coverage = (# concepts in top K results) / (# concepts in ground truth answer)
```
Measures: Does the system retrieve all necessary concepts to answer the question?

**Multi-Hop Retrieval Accuracy**
```
Accuracy = (# correct multi-hop paths found) / (# ground truth paths)
```
Measures: Can the system chain concepts together (A→B→C)?

---

## 2. Gravity Field Quality Metrics

### 2.1 Clustering Metrics

**Silhouette Score**
```
s(i) = (b(i) - a(i)) / max(a(i), b(i))
where a(i) = avg distance to same cluster
      b(i) = avg distance to nearest other cluster
Range: [-1, 1], higher is better
```
Measures: How well-separated are conceptual clusters?

**Modularity**
```
Q = (1/2m) × Σ[A_ij - (k_i × k_j)/2m] × δ(c_i, c_j)
```
Measures: Strength of division into clusters (used in network analysis)

### 2.2 Attraction/Repulsion Behavior

**Positivity Attraction Score**
```
For statement "A is B":
  Score = similarity(A, B) after adding statement / similarity before
  Expected: > 1.0 (concepts moved closer)
```

**Negation Repulsion Score**
```
For statement "A is NOT B":
  Score = similarity(A, B) after adding statement / similarity before
  Expected: < 1.0 (concepts moved apart)
```

**Repulsion Strength**
```
Repulsion = Δ distance / baseline distance
Expected: Positive for negations, proportional to statement strength
```

---

## 3. Temporal Dynamics Metrics

### 3.1 Decay Behavior

**Forgetting Curve Fit**
```
R(t) = e^(-t/τ)  (exponential decay)
or R(t) = t^(-α)  (power law)

Measure: How well does mass decay fit expected curves?
```

**Half-Life**
```
t_1/2 = time for concept mass to reduce to 50% of original
```
Measures: How quickly unreinforced concepts fade

### 3.2 Reinforcement Effectiveness

**Reinforcement Gain**
```
Gain = (mass after N reinforcements) / (baseline mass)
```
Expected: Proportional to reinforcement frequency

**Staleness Impact**
```
Impact = Δ(centroid distance) / staleness
```
Measures: How far concepts drift per unit of staleness

---

## 4. Knowledge Graph Comparison Metrics

### 4.1 Against Traditional KG

**Edge Precision**
```
For each KG edge (A, relation, B):
  Check if similarity(A, B) in holographic system > threshold
  
Precision = (# confirmed edges) / (# KG edges)
```

**Edge Recall**
```
For each strong connection in holographic system:
  Check if exists as edge in KG
  
Recall = (# found edges) / (# strong connections)
```

### 4.2 Contradiction Handling

**Ambiguity Resolution**
```
Given: "A is B" and "A is NOT B"
  
Traditional KG: Must choose one or version (0/1 decision)
Holographic: Both coexist with tension (continuous)

Metric: Can the system retrieve both perspectives when queried?
```

---

## 5. Efficiency Metrics

### 5.1 Computational Cost

**Query Latency**
```
Latency = time to retrieve top-K results
```
Target: < 100ms for typical queries

**Indexing Throughput**
```
Throughput = # concepts added per second
```

**Memory Usage**
```
Usage = total bytes / # concepts
```

### 5.2 Scaling Behavior

**Query Time vs Dataset Size**
```
Plot: log(query_time) vs log(dataset_size)
Expected: FAISS provides sub-linear scaling
```

---

## 6. Benchmark Datasets

### 6.1 Synthetic Datasets

**Cluster Quality Test**
```
Create: 5 clusters of 20 concepts each
  - Cluster 1: Physics concepts
  - Cluster 2: Biology concepts
  - Cluster 3: History concepts
  - Cluster 4: Art concepts
  - Cluster 5: Math concepts

Measure: Silhouette score, within-cluster similarity
```

**Negation Test**
```
Add pairs:
  ("A is B", "A is NOT B")
  ("X causes Y", "X does NOT cause Y")

Measure: Repulsion score, distance increase
```

**Temporal Test**
```
Add concepts with controlled reinforcement patterns:
  - High frequency (every 2 steps)
  - Medium frequency (every 10 steps)
  - Low frequency (once, never reinforced)

Measure: Mass over time, distance from centroid
```

### 6.2 Real-World Datasets

**ConceptNet** (existing KG)
- 8M+ relationships
- Test: Can holographic system discover similar relationships?

**Wikipedia Abstracts**
- Extract facts from science/history articles
- Test: Query answering, concept retrieval

**FEVER** (Fact Extraction and VERification)
- Claims + evidence
- Test: Can system verify claims using stored knowledge?

---

## 7. Example Evaluation Suite

```python
# Pseudocode for evaluation

def evaluate_retrieval(hg, test_queries):
    """Measure retrieval quality."""
    precisions, recalls, mrrs = [], [], []
    
    for query, ground_truth in test_queries:
        results = hg.search_text(query, top_k=10)
        
        relevant = [r for r in results if r.id in ground_truth]
        precisions.append(len(relevant) / 10)
        recalls.append(len(relevant) / len(ground_truth))
        
        # MRR
        for i, r in enumerate(results):
            if r.id in ground_truth:
                mrrs.append(1 / (i + 1))
                break
    
    return {
        'precision@10': np.mean(precisions),
        'recall@10': np.mean(recalls),
        'MRR': np.mean(mrrs)
    }

def evaluate_gravity_field(hg, test_triplets):
    """Measure gravity field behavior."""
    attraction_scores = []
    repulsion_scores = []
    
    for stmt_type, concept_a, concept_b, relation in test_triplets:
        # Measure before
        sim_before = hg.similarity(concept_a, concept_b)
        
        # Add statement
        if stmt_type == "positive":
            hg.add_text(f"{concept_a} {relation} {concept_b}")
            expected = "attraction"
        else:  # negative
            hg.add_text(f"{concept_a} is NOT {relation} {concept_b}")
            expected = "repulsion"
        
        # Measure after
        sim_after = hg.similarity(concept_a, concept_b)
        change = sim_after / sim_before
        
        if expected == "attraction":
            attraction_scores.append(change)
        else:
            repulsion_scores.append(change)
    
    return {
        'attraction_score': np.mean(attraction_scores),  # > 1.0
        'repulsion_score': np.mean(repulsion_scores),    # < 1.0
    }

def evaluate_temporal_dynamics(hg, n_steps=100):
    """Measure decay and reinforcement."""
    # Add concepts with known reinforcement patterns
    freq_id = hg.add_text("Frequent concept")
    rare_id = hg.add_text("Rare concept")
    
    masses_freq = []
    masses_rare = []
    
    for step in range(n_steps):
        if step % 5 == 0:
            hg.add_text("Frequent concept")  # Reinforce
        
        # Add noise
        hg.add_text(f"Noise {step}")
        hg.decay(steps=1)
        
        masses_freq.append(hg.store.sim.concepts[freq_id].mass)
        masses_rare.append(hg.store.sim.concepts[rare_id].mass)
    
    # Fit decay curve for rare concept
    from scipy.optimize import curve_fit
    def exp_decay(t, tau):
        return np.exp(-t / tau)
    
    tau, _ = curve_fit(exp_decay, range(n_steps), masses_rare)
    
    return {
        'final_mass_frequent': masses_freq[-1],
        'final_mass_rare': masses_rare[-1],
        'decay_time_constant': tau[0]
    }
```

---

## 8. Baseline Comparisons

Compare against:

1. **Pure Vector DB** (Pinecone, FAISS only)
   - No gravity field
   - No decay
   - Measure: retrieval quality difference

2. **Traditional KG** (Neo4j + embeddings)
   - Structured edges
   - No natural decay
   - Measure: query answering accuracy

3. **RAG System** (Vector DB + LLM)
   - Retrieval-Augmented Generation
   - Measure: answer quality, latency

---

## 9. Reporting Template

```markdown
## Evaluation Results

### Dataset
- Name: [ConceptNet / Wikipedia / Custom]
- Size: [# concepts, # statements]
- Domain: [Physics / General / Multi-domain]

### Retrieval Quality
- Precision@10: 0.XX
- Recall@10: 0.XX
- MRR: 0.XX
- NDCG@10: 0.XX

### Gravity Field Quality
- Silhouette Score: 0.XX
- Attraction Score: X.XX (> 1.0 ✓)
- Repulsion Score: 0.XX (< 1.0 ✓)

### Temporal Dynamics
- Decay Half-Life: XX steps
- Reinforcement Gain: X.XX
- Forgetting Curve Fit: R² = 0.XX

### Efficiency
- Avg Query Latency: XX ms
- Index Throughput: XX concepts/sec
- Memory per Concept: XX bytes

### vs Baselines
- Vector DB: [better/worse/same] by XX%
- Traditional KG: [better/worse/same] by XX%
- RAG System: [better/worse/same] by XX%
```

---

## 10. Future Metrics

As the system evolves, consider:

- **Analogical reasoning**: Can it find "A is to B as C is to ___"?
- **Contradiction detection**: How well does it identify conflicting statements?
- **Uncertainty quantification**: Does similarity score correlate with answer confidence?
- **Transfer learning**: Can knowledge from one domain help in another?
- **Interpretability**: Can we visualize and explain why concepts are connected?

---

## Summary

Key metrics to track:

1. **Retrieval**: Precision, Recall, MRR, NDCG
2. **Gravity**: Clustering quality, attraction/repulsion scores
3. **Temporal**: Decay curves, reinforcement effectiveness
4. **Efficiency**: Latency, throughput, memory
5. **Comparison**: vs Vector DB, KG, RAG systems

These metrics enable:
- Systematic evaluation
- Comparison with baselines
- Tracking improvements over time
- Scientific validation of the approach
