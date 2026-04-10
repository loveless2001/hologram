# KG + Drift Pivot Resume Notes

Last updated: 2026-03-03

## Goal
Pivot Hologram toward a drift-detection-first architecture with optional retrieval coupling.

Primary use cases:
- Cross-time concept/domain drift
- LLM response drift for same prompt across models, versions, and tuning states
- Human-readable semantic analysis via per-batch knowledge graphs

## Implemented in This Session

### New modules
- `hologram/kg/`
  - `models.py`: `KGNode`, `KGEdge`, `BatchKGSnapshot`
  - `builder.py`: `build_batch_kg_snapshot(batch_id, items, ...)`
- `hologram/drift/`
  - `models.py`: `DriftComparisonInput`, `DriftDimension`, `DriftReport`
  - `detectors.py`:
    - `embedding_centroid_drift(...)`
    - `kg_structure_drift(...)`
  - `engine.py`: `compare_batches(...)`

### API integration
- Added to `Hologram` (`hologram/api.py`):
  - `build_kg_batch(...)`
  - `compare_drift(...)`
- Added server endpoints (`hologram/server.py`):
  - `POST /kg/build_batch`
  - `POST /drift/compare`

### Test coverage
- `tests/test_kg_drift_engine.py`
  - validates batch KG build
  - validates drift compare output dimensions

Test command used:
```bash
./.venv/bin/pytest -q tests/test_kg_drift_engine.py
```

## Current Behavior

### KG builder
Input items are flexible; minimal usable field is `payload` (fallback `text`).
Other fields (`id`, `item_type`, `domain`, `model`, etc.) are optional.

Graph extraction currently produces:
- concept nodes from `extract_concepts(...)`
- co-occurrence edges (`relation = "co_occurs"`)

### Drift engine
Current dimensions:
- `embedding_centroid`: centroid distance between baseline and target text embeddings
- `kg_structure`: node/edge churn between baseline and target batch snapshots

Overall score:
- weighted aggregate: embedding 0.55, KG 0.45

## Quick API Examples

### Build batch KG
```bash
curl -X POST "http://localhost:8000/kg/build_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "demo",
    "batch_id": "batch:v1",
    "items": [
      {"id": "q1", "item_type": "query", "payload": "Explain gravity"},
      {"id": "r1", "item_type": "response", "payload": "Gravity is spacetime curvature"}
    ]
  }'
```

### Compare drift
```bash
curl -X POST "http://localhost:8000/drift/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "demo",
    "baseline_id": "modelA:v1",
    "target_id": "modelB:v1",
    "baseline_items": [{"id": "rA", "payload": "Gravity is a force."}],
    "target_items": [{"id": "rB", "payload": "Gravity is spacetime curvature."}]
  }'
```

## Known Gaps / Next Work

1. Typed KG relations are not implemented yet.
- Add relation extraction for `supports`, `contradicts`, `causes`, `defines`.

2. Response-specific drift dimensions are minimal.
- Add style/factual/claim/polarity/stability dimensions.

3. Persistence for snapshots/reports is not yet first-class.
- Add SQLite tables or a dedicated drift store API.

4. Calibration/controls are not yet implemented.
- Add fixed sentinel prompt set to detect instrumentation drift.

5. Multi-embedder agreement is not yet implemented.
- Add 2-3 embedding views and confidence uplift only on agreement.

## Resume Checklist
When resuming, start here:
1. Implement typed relation extraction in `hologram/kg/builder.py` (or split into `extract.py` + `relations.py`).
2. Add response drift detectors in `hologram/drift/detectors.py`.
3. Add persistence layer for snapshots and reports.
4. Add endpoint `POST /drift/response/record` for longitudinal experiments.
5. Add regression tests for detector correctness and schema stability.
