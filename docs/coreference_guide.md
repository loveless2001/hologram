# Coreference Resolution in Hologram

## Overview

Hologram uses a **hybrid coreference resolution system** that combines neural network-based structural resolution with physics-based vector attraction for ambiguous cases.

## Architecture

### Two-Stage Pipeline

```
Text Input
    ↓
┌─────────────────────────────────────┐
│ Stage 1: FastCoref (Neural)         │
│ - Resolves pronouns structurally    │
│ - Maps: "It" → "The engine"         │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: Gravity Fallback (Vector)  │
│ - Handles deictics ("this", "that") │
│ - Uses mass-weighted cosine sim     │
└─────────────────────────────────────┘
    ↓
Resolved Text + Coref Map
```

## Components

### 1. `hologram/coref.py`

**Primary Resolver**: Uses `fastcoref` library for high-accuracy structural coreference.

```python
from hologram.coref import resolve

text = "The fusion engine is unstable. It requires cooling."
resolved_text, coref_map = resolve(text)

# resolved_text: "The fusion engine is unstable. The fusion engine requires cooling."
# coref_map: {'It': 'The fusion engine'}
```

**Implementation Details**:
- Singleton pattern for model efficiency
- Lazy loading (model loads on first use)
- Returns both resolved text and pronoun→antecedent map
- Handles cluster-based resolution (first mention = antecedent)

### 2. `hologram/gravity.py`

**Fallback Resolver**: Vector-based resolution using the gravity field.

```python
# In Gravity class
def resolve_pronoun(self, sentence: str, pronoun_span: str) -> Optional[str]:
    """
    Resolve a pronoun using vector similarity in the gravity field.
    Fallback mechanism when structural coref fails.
    """
```

**Algorithm**:
1. Encode the sentence context as a vector
2. Find nearest concept in the gravity field
3. Weight by mass: `score = cosine_sim * log(1 + mass)`
4. Return concept with highest score

**Use Cases**:
- Deictic references ("this", "that", "these", "those")
- Abstract clause references ("This solved the problem")
- Low-confidence FastCoref predictions
- Short/ambiguous sentences

### 3. `hologram/store.py`

**Trace Storage**: Extended `Trace` dataclass to store resolution metadata.

```python
@dataclass
class Trace:
    trace_id: str
    kind: str
    content: str              # Original text
    vec: np.ndarray
    meta: dict
    resolved_text: Optional[str] = None      # NEW
    coref_map: Optional[Dict[str, str]] = None  # NEW
```

### 4. `hologram/api.py`

**Pipeline Integration**: Automatic resolution in `Hologram.add_text()`.

```python
def add_text(self, glyph_id: str, text: str, ...):
    # 1. Structural Resolution
    resolved_text, coref_map = resolve(text)
    
    # 2. Gravity Fallback (if enabled)
    if Config.coref.ENABLE_GRAVITY_FALLBACK and self.field:
        for pronoun in ["this", "that", "it", "these", "those"]:
            if pronoun in text and pronoun not in coref_map:
                antecedent = self.field.resolve_pronoun(text, pronoun)
                if antecedent:
                    coref_map[pronoun] = antecedent
    
    # 3. Use resolved text for concept extraction
    concepts = extract_concepts(resolved_text)
    
    # 4. Store both original and resolved
    trace = Trace(
        content=text,              # Original
        resolved_text=resolved_text,
        coref_map=coref_map
    )
```

## Configuration

### `hologram/config.py`

```python
@dataclass
class CorefConfig:
    ENABLE_COREF: bool = True
    ENABLE_GRAVITY_FALLBACK: bool = True
    COREF_MODEL: str = "fastcoref"
```

### Usage

```python
from hologram.config import Config

# Disable coreference entirely
Config.coref.ENABLE_COREF = False

# Use only FastCoref (no gravity fallback)
Config.coref.ENABLE_GRAVITY_FALLBACK = False
```

## Benefits

### 1. Improved Concept Extraction

**Before**:
```
Input: "The engine failed. It was overheating."
GLiNER Output: ['engine', 'failed', 'It', 'overheating']
Problem: "It" extracted as separate concept
```

**After**:
```
Input: "The engine failed. It was overheating."
Resolved: "The engine failed. The engine was overheating."
GLiNER Output: ['engine', 'failed', 'engine', 'overheating']
Result: Proper concept reinforcement (mass += 1)
```

### 2. Reduced Concept Fragmentation

Pronouns no longer create orphan concepts in the gravity field.

### 3. Better Memory Reconstruction

When retrieving memories, the system can show:
- Original text (for human readability)
- Resolved text (for semantic analysis)
- Coref map (for explainability)

## Performance

### FastCoref
- **Model size**: ~400MB (downloads on first use)
- **Inference time**: ~0.5-2s per document (CPU)
- **Accuracy**: High for standard pronouns (he, she, it, they)

### Gravity Fallback
- **Computation**: O(N) where N = number of concepts
- **Latency**: <10ms for typical fields (<1000 concepts)
- **Use case**: Deictics and ambiguous references

## Testing

See `tests/test_coref.py` for comprehensive test coverage:

1. **Basic Resolution**: Verifies FastCoref resolves "It" → "engine"
2. **Gravity Fallback**: Tests vector-based resolution for "This"
3. **Integration**: End-to-end test with `Hologram.add_text()`
4. **No False Fusions**: Ensures pronouns don't pollute concept space

## Limitations

### Current
1. **English-only**: FastCoref is trained on English corpora
2. **Computational cost**: Neural model adds latency to ingestion
3. **Fallback simplicity**: Gravity fallback uses naive pronoun detection

### Future Improvements
1. **Multilingual support**: Use mCoref or XLM-based models
2. **Confidence scoring**: Skip low-confidence resolutions
3. **Contextual fallback**: Use sentence embeddings instead of full text
4. **Incremental resolution**: Resolve only new text in updates

## Examples

### Example 1: Technical Documentation

```python
h = Hologram.init()

text = """
The fusion reactor uses magnetic confinement. 
It requires superconducting coils. 
These generate a toroidal field.
This stabilizes the plasma.
"""

trace_id = h.add_text("reactor_doc", text)
trace = h.store.get_trace(trace_id)

print(trace.coref_map)
# {
#   'It': 'The fusion reactor',
#   'These': 'superconducting coils',
#   'This': 'toroidal field'  # (via gravity fallback)
# }
```

### Example 2: Narrative Text

```python
text = "Alice met Bob. She gave him a book."

resolved, coref_map = resolve(text)
# resolved: "Alice met Bob. Alice gave Bob a book."
# coref_map: {'She': 'Alice', 'him': 'Bob'}
```

## Debugging

### Enable Logging

```python
import logging
logging.basicConfig(level=logging.INFO)

# You'll see:
# [COREF] resolved 4 pronouns
# [COREF-FALLBACK] 'This' → 'fusion engine' (score: 0.85)
```

### Inspect Trace

```python
trace = h.store.get_trace(trace_id)
print(f"Original: {trace.content}")
print(f"Resolved: {trace.resolved_text}")
print(f"Map: {trace.coref_map}")
```

## Design Principles

1. **Graceful Degradation**: If FastCoref fails, fall back to gravity
2. **Explainability**: Store both original and resolved text
3. **Non-invasive**: Can be disabled via config
4. **Physics-first**: Gravity fallback uses existing vector field
5. **No Symbolic Logic**: Avoid brittle rule-based systems
