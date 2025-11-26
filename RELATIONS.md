# Concept Relations Feature

## Overview

The `/search` endpoint now returns **explicit concept relations** extracted from the gravity field, showing how concepts are interconnected.

## API Response Example

```json
{
  "query": "speed of light",
  "results": [
    {
      "content": "speed of light",
      "score": 1.0,
      "relations": [
        {
          "concept": "spacetime",
          "strength": 0.85
        },
        {
          "concept": "Special Relativity",
          "strength": 0.78
        }
      ]
    }
  ]
}
```

## How It Works

### 1. Gravity Field Relations
When concepts are added to the holographic memory, the gravity field computes **pairwise relations** based on:
- Vector similarity (cosine distance)
- Co-occurrence patterns
- Mutual influence during drift

These are stored in:
```python
gravity.relations: Dict[Tuple[str, str], float]
```

### 2. Relation Strength
- **Range**: -1.0 to 1.0
- **Positive values**: Concepts attract each other (co-occur, semantically similar)
- **Negative values**: Concepts repel (e.g., negation: "not gravity" vs "gravity")
- **Magnitude**: Indicates strength of relationship

### 3. Search Enhancement
The `/search` endpoint now:
1. Finds semantic matches (standard vector search)
2. For each match, extracts its relations from `gravity.relations`
3. Returns top 5 strongest relations (sorted by absolute strength)

## Usage

### Via API
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "speed of light", "top_k": 5}'
```

### Via Streamlit UI
1. Load a KB (e.g., `relativity.txt`)
2. Go to "🔍 Semantic Search" tab
3. Search for a keyword
4. Click "🔗 X related concepts" to expand relations

### Via Python
```python
import requests

response = requests.post("http://localhost:8000/search", json={
    "query": "time dilation",
    "top_k": 5
})

for result in response.json()["results"]:
    print(f"{result['content']} (score: {result['score']})")
    if result.get('relations'):
        for rel in result['relations']:
            print(f"  → {rel['concept']} (strength: {rel['strength']})")
```

## Interpretation

### High Relation Strength (>0.7)
- Concepts frequently co-occur in the same context
- Strong semantic similarity
- Likely to be part of the same knowledge domain

### Medium Relation Strength (0.4-0.7)
- Moderate association
- May appear in related but distinct contexts
- Could be tangentially related

### Low Relation Strength (<0.4)
- Weak or no meaningful relationship
- Possibly noise from small datasets
- Should be filtered in production use

### Negative Relation Strength (<0)
- Concepts are semantically opposed
- One negates or contradicts the other
- Example: "gravity exists" vs "gravity doesn't exist"

## Current Limitations

### 1. Small Dataset Effect
With limited concepts (~20 in test KB), relation strengths tend to cluster around **~0.5** because:
- All concepts were added in a short timeframe
- Limited opportunities for differential reinforcement
- Gravity field hasn't had time to separate domains

**Solution**: Larger KBs with diverse topics will show clearer relation patterns.

### 2. Temporal Bias
All concepts have similar relation strengths because they were added simultaneously.

**Future enhancement**: Temporal decay and reinforcement-based relation weighting.

### 3. Relation Type
Currently, only **strength** is returned, not **type** (e.g., "part-of", "similar-to", "causes").

**Future enhancement**: Classify relation types using:
- Extracted verbs from GLiNER
- Co-occurrence patterns
- Sentence structure analysis

## Best Practices

### For Testing
```python
# Use diverse KB with clear domains
kb_content = """
Physics: Einstein developed relativity theory.
Physics: Gravity affects spacetime curvature.
Biology: DNA contains genetic information.
Biology: Cells are the basic units of life.
"""
# Relations within domains should be stronger than across domains
```

### For Production
1. **Filter weak relations**: Only show relations with |strength| > 0.6
2. **Limit results**: Return top 3-5 relations per concept
3. **Add context**: Explain why concepts are related (work in progress)
4. **Temporal weighting**: Boost recent relations

## Examples

### Strong Relations (Expected)
```
"Special Relativity" → "time dilation" (0.85)
"DNA" → "genetic information" (0.92)
"gravity" → "spacetime" (0.78)
```

### Weak Relations (Noise)
```
"baseball" → "quantum mechanics" (0.12)
"apple pie" → "black hole" (0.05)
```

### Negative Relations
```
"gravity exists" → "gravity doesn't exist" (-0.85)
"light" → "darkness" (-0.42)
```

## Testing

Run the test script:
```bash
python test_search_relations.py
```

Expected output shows:
- Each search result with its similarity score
- Top 5 related concepts with strengths
- Comparison across different queries

## Future Enhancements

### Short-term
- [ ] Filter relations by minimum strength threshold
- [ ] Add relation type classification
- [ ] Visualize relation graph in UI

### Medium-term
- [ ] Temporal decay for relation strengths
- [ ] Context-aware relations (sentence-level)
- [ ] Bidirectional relation explanations

### Long-term
- [ ] Graph-based navigation (explore concept networks)
- [ ] Automatic relation type inference
- [ ] Multi-hop relation discovery (A→B→C)

## Graph-Based Reconstruction

The system now leverages these relations for **Knowledge Reconstruction**:
1.  **Subgraph Retrieval**: Instead of just lists, we retrieve a structured subgraph (nodes + edges).
2.  **LLM Synthesis**: The LLM uses this graph to generate coherent explanations.

## Contextual Disambiguation (Mitosis)

Relations play a key role in **Concept Mitosis**:
- **Tension Detection**: If a concept has strong relations to two distinct, separated clusters, it is flagged for splitting.
- **Bridge Links**: When split, a weak relation (~0.15) is maintained between siblings to allow cross-domain traversal.

---

**Last Updated**: 2023-11-26
**Feature Status**: ✅ Implemented (Beta)
