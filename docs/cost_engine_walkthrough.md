# Cost Engine Implementation Walkthrough

## Overview
The Cost Engine is a diagnostic meta-layer added to the Hologram system to measure "cognitive effort" metrics (Resistance, Entropy, Drift Cost) without altering the underlying concept vectors. It provides insights into the stability and coherence of the memory field.

## Key Components

### 1. `hologram/cost_engine.py`
This is the core module containing:
- **`CostEngineConfig`**: Configuration dataclass with presets (`analytical`, `creative`, `conservative`).
- **`CostReport`**: Dataclass for structured reporting of metrics and suggestions.
- **`CostEngine`**: The main class that calculates metrics and generates suggestions.

### 2. `hologram/gravity.py` Integration
- **`Concept` Dataclass**: Updated to include `previous_vec` field to track vector drift over time.
- **`Gravity.add_concept`**: Updated to store `previous_vec` when updating existing concepts.

### 3. `hologram/config.py` Integration
- The system now uses a centralized configuration file.
- `GravityConfig` in `config.py` controls thresholds for mitosis and fusion, which influence the Cost Engine's suggestions.

## Metrics Explained

1.  **Resistance (R)**: Measures how hard it is to integrate a new concept.
    - High R: Concept is contradictory or distant from existing knowledge.
    - Low R: Concept fits well with existing knowledge.
    - Formula: `1.0 - mean_similarity(concept, neighbors)`

2.  **Entropy (S)**: Measures the disorder of the local neighborhood.
    - High S: Neighborhood is scattered/diverse (high cognitive load).
    - Low S: Neighborhood is tight/coherent.
    - Formula: Shannon entropy of the similarity distribution of neighbors.

3.  **Drift Cost (D)**: Measures how much the field has shifted.
    - High D: Significant semantic shift (instability).
    - Low D: Stable field.
    - Formula: Euclidean distance between `previous_vec` and `current_vec`.

4.  **Total Cost (C)**: Weighted sum of R, S, and D.
    - `C = w_r * R + w_s * S + w_d * D`

## Suggestions

Based on the Total Cost and individual metrics, the engine suggests actions:
- **`split`**: If Entropy is high (ambiguity).
- **`fuse`**: If Resistance is low (redundancy).
- **`stabilize`**: If Drift is high (instability).
- **`no-action`**: If cost is low (equilibrium).

## Usage Example

```python
from hologram.cost_engine import CostEngine, CostEngineConfig
from hologram.api import Hologram

# Initialize
holo = Hologram.init(use_gravity=True)
engine = CostEngine(holo.field.sim, config=CostEngineConfig.preset("analytical"))

# ... add concepts ...

# Evaluate a concept
report = engine.evaluate_concept("quantum_field")
print(f"Total Cost: {report.total_cost:.3f}")
print(f"Suggestion: {report.suggestion}")
```

## Testing
Unit tests are located in `tests/test_cost_engine.py` and cover:
- Metric calculations (Resistance, Entropy, Drift).
- Suggestion logic for different presets.
- Integration with the Gravity field.
