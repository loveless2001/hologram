---
title: "Gravity Field Performance Optimization"
description: "Implement FAISS index, probe caching, and dynamic quantization for faster probes"
status: complete
priority: P1
effort: 3h
branch: main
tags: [performance, gravity, faiss, optimization]
created: 2026-02-05
---

# Gravity Field Performance Optimization

## Overview

Optimize gravity field probe performance through three key improvements:
1. **A1**: Persistent FAISS index for `nearest_concepts()`
2. **A3**: Probe neighbor caching to eliminate repeated searches
3. **A4**: Dynamic quantization calibration based on dataset scale

**Source**: [Brainstorm Report](../reports/brainstorm-260205-1530-gravity-field-optimization.md)

## Success Criteria

- Probe latency <50ms at 100K concepts
- Graceful degradation to 1M concepts
- No regression in fusion/mitosis behavior
- Minimal API changes (internal optimization)

## Implementation Phases

| Phase | Description | Status | Effort |
|-------|-------------|--------|--------|
| [Phase 1](phase-01-persistent-faiss-index.md) | Persistent FAISS index for nearest_concepts() | ✅ complete | 1.5h |
| [Phase 2](phase-02-probe-neighbor-caching.md) | Probe neighbor caching for fast retrieval | ✅ complete | 1h |
| [Phase 3](phase-03-dynamic-quantization.md) | Dynamic quantization by scale | ✅ complete | 30m |

## Target Files

- `hologram/gravity.py` (primary)
- `tests/test_probe_drift.py` (performance tests)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Gravity Class                            │
├─────────────────────────────────────────────────────────────┤
│  NEW: _faiss_index (IndexFlatIP)                           │
│  NEW: _index_names (List[str])                              │
│  NEW: _index_dirty (bool)                                   │
├─────────────────────────────────────────────────────────────┤
│  _rebuild_index()     <- Lazy rebuild under lock            │
│  nearest_concepts()   <- FAISS search + mass rerank         │
│  run_probe()          <- Pre-fetch neighbors, use cache     │
│  calibrate_quantization() <- Scale-aware threshold          │
└─────────────────────────────────────────────────────────────┘
```

## Dependencies

- FAISS already imported in gravity.py
- No new dependencies required

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Index rebuild latency | Medium | Lazy rebuild, only when dirty |
| Cache staleness | Low | Per-probe instance cache, short-lived |
| Memory pressure | Low | IndexFlatIP is O(N×D×4 bytes) |
