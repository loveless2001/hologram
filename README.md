
# Holographic Memory (Minimal Starter)

A tiny, hackable starter kit that shows how to use **glyphs as anchors** over multi‑modal traces
(tokens, text chunks, images) and back them with simple embeddings. The goal is to keep code small and clear.

**Key idea:** Don't index everything flat. Create *glyphs* (symbolic anchors) and attach traces under them.
Retrieve via glyph → then pull associated traces (and their vectors).

This repo avoids heavy deps: it uses NumPy and a tiny in‑memory cosine index.
Swap `SimpleIndex` with FAISS/ScaNN later if you need scale.

## Layout
- `hologram/config.py` – knobs & vector dims
- `hologram/embeddings.py` – pluggable embedding adapters (toy hashing model included)
- `hologram/store.py` – in‑memory registry + vector index (replace with DB + FAISS later)
- `hologram/glyphs.py` – glyph registry & linking traces
- `hologram/api.py` – simple write/read operations
- `demo.py` – runnable demo

## Run
```bash
python demo.py
```

Expected: it will create a few glyphs, attach text/image traces, and run example queries.

## Upgrade Path (when you're ready)
- Replace `SimpleIndex` with FAISS + HNSW and persist vectors to disk.
- Store glyphs/traces in SQLite or Postgres (see `store.py` for the in‑mem schema).
- Add a CLIP/Image encoder in `embeddings.py` for image vectors.
- Add a RAG step after recall for synthesis (e.g., call your LLM with the retrieved traces).
