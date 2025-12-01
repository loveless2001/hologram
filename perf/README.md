
# Hologram Performance Tracking

## Baselines

### v0 (2025-12-01)
- **Ingest**: ~0.24s/line (Hasher) -> ~1.0s/line (MiniLM + Gravity Vectorized)
- **Search**: < 1ms (FAISS)
- **Viz**: < 5ms (PCA Cached)

## Benchmarks
Run the following scripts to profile:

```bash
python scripts/profile_ingest.py
python scripts/profile_search.py
```

## Regression Tests
Run with pytest:

```bash
pytest tests/test_perf_baseline.py
```
