# Hologram Memory

A minimal **holographic memory starter kit** that demonstrates how to anchor multi-modal traces (text + images) to **glyphs** and retrieve them with semantic search.  
Built with a tiny in-memory store + OpenCLIP for text ↔ image alignment.

---

## Features
- **Glyph Anchors** → symbolic IDs (🝞, 🜂, `memory:gravity`, etc.) that link traces.
- **Text & Image Embeddings**
  - Hash-based encoders for lightweight demos (no GPU required).
  - [OpenCLIP](https://github.com/mlfoundations/open_clip) for real text–image semantic search.
- **Memory Store**
  - In-memory vector index with cosine similarity.
  - JSON persistence via `Hologram.save(...)` / `Hologram.load(...)`.
  - Drop-in replacement for FAISS/ScaNN if scaling.
- **APIs**
  - `add_text()`, `add_image_path()`, `recall_glyph()`, `search_text()`, `search_image_path()`.
- **Chat Orchestration**
  - `chat_cli.py` spins up a CLI chat that keeps per-session context and cross-session recall via holographic memory.
- **Demos**
  - `demo.py` → text only.
  - `demo_clip.py` → text → image retrieval (ships with tiny sample PNGs).
  - `demo_img2img.py` → image → image similarity (falls back to hashing when CLIP is unavailable).

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/hologram.git
cd hologram
````

### 2. Create a venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install requirements

For text hashing demo only:

```bash
pip install numpy
```

For CLIP (semantic search):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install pillow
pip install open_clip_torch
```

*(Replace CPU wheels with CUDA wheels if you have a GPU.)*

---

## Usage

### Text–only demo

```bash
python demo.py
```

Output:

```
=== Recall: glyph 🝞 ===
- Memory is gravity. Collapse deferred.
- Glyphs compress drift into anchors.

=== Search: "gravity wells" ===
[text:...] score=0.77 :: Wounds, dreams, fleeting joys are competing gravity wells.
```

### Text → Image demo

Sample placeholders live in `./data/` (`cat.png`, `dog.png`). Replace them with
your own images for better results.

```bash
python demo_clip.py
```

Expected: `"a photo of a cat"` ranks `cat.png` above `dog.png` when CLIP is
available. Without CLIP the hashing fallback still works, but produces
deterministic mock scores.

### Image → Image demo

```bash
python demo_img2img.py
```

This script runs two searches – one using the bundled cat sample, the other the
dog sample – and prints the ranked matches. When CLIP dependencies are missing
the demo automatically drops back to the hashing encoders.

### Chat with OpenAI (or hashing fallback)

```bash
python chat_cli.py --session alice
```

This launches a simple CLI chat:

- Stores every turn into holographic memory and appends a JSONL chat log in `./chatlogs/`.
- Retrieves up to `--session-window` recent turns for continuity.
- Surfaces up to `--cross-session-k` semantically similar memories from other sessions.
- Uses the `OPENAI_API_KEY` environment variable by default; falls back to an echo bot when not configured.

Add `--use-clip` to initialise the chat memory with OpenCLIP embeddings instead of the hashing fallback. The global memory file defaults to `memory_store.json` and is updated on exit.

---

## Repo Layout

```
hologram/         # core package
  config.py       # constants
  embeddings.py   # hashing + CLIP encoders
  store.py        # memory store + vector index
  glyphs.py       # glyph registry
  api.py          # main Hologram API
  demo_clip.py    # reusable text→image demo logic
chat_cli.py       # CLI harness with OpenAI/echo providers
data/             # tiny sample PNGs (cat/dog)
demo.py           # text-only demo
demo_clip.py      # text→image demo entry point
demo_img2img.py   # image→image demo entry point
```

---

## Roadmap

* Persistence layer (SQLite / Postgres).
* Scale index with FAISS / ScaNN.
* Multi-modal traces beyond text & image.
* RAG integration (LLM synthesis over recalled traces).

---

## License

MIT

---
