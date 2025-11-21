# Hologram Memory

A holographic memory sandbox that anchors multi-modal traces to glyphs, stores them in a lightweight vector index, and experiments with “memory gravity” fields to model concept drift.

---

## Highlights
- Glyph anchors (`🝞`, `🜂`, `memory:gravity`, …) collect related traces across sessions.
- Dual encoder stack: hashing fallback works everywhere; OpenCLIP adds semantic text ↔ image retrieval.
- **Negation-aware** gravity field powered by FAISS: positive statements attract concepts, negative statements repel them.
- **Reinforcement-based decay**: unreinforced concepts drift to memory periphery and lose influence over time.
- JSON-backed memory store with glyph registry, summarisation helpers, and **optimized state persistence** (saves final vector state to avoid costly replay).
- Chat orchestration (`chat_cli.py`) with session logs plus a tiny FastAPI surface for programmatic access.

---

## Components
- `hologram/api.py` – public `Hologram` API (`add_text`, `add_image_path`, search, persistence).
- `hologram/chatbot.py` – chat memory, provider abstractions, CLI orchestration helpers.
- `hologram/gravity.py` – concept drift simulation and FAISS-backed `GravityField` (supports state export/import).
- `hologram/embeddings.py` – hashing encoders and CLIP wrappers (text and image).
- `chat_cli.py` – command-line chat demo that keeps cross-session context.
- `api_server/main.py` – FastAPI service exposing `/add_text` + `/search` endpoints.
- `demo.py`, `demo_clip.py`, `demo_img2img.py` – runnable examples (text-only, text→image, image→image).
- `tests/test_chatbot.py` – regression coverage for the chat memory + persistence path.

---

## Installation

```bash
git clone https://github.com/loveless2001/hologram.git
cd hologram
python3 -m venv .venv
source .venv/bin/activate
```

### Core dependencies

`GravityField` imports FAISS; install it even when using the hashing encoders:

```bash
pip install numpy faiss-cpu Pillow
```

### Optional dependencies

For the API server:
```bash
pip install fastapi uvicorn
```

For the Chatbot (OpenAI provider):
```bash
pip install openai
```

For CLIP embeddings (semantic image search):
```bash
pip install torch torchvision open_clip_torch
```

If FAISS wheels are not available on your platform, initialise with `Hologram.init(use_gravity=False)`.

### Optional extras
- CLIP stack (semantic text ↔ image):  
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`  
  `pip install pillow open_clip_torch`
- Chat providers: `pip install openai`
- REST API: `pip install fastapi uvicorn[standard]`
- Development helpers: `pip install pytest`

*(Switch to CUDA wheels if you have a GPU.)*

---

## Quickstart (Python)

```python
from hologram import Hologram

holo = Hologram.init(use_clip=False)  # add use_gravity=False if FAISS is unavailable
holo.glyphs.create("🝞", title="Curvature Anchor")

trace_id = holo.add_text("🝞", "Memory is gravity. Collapse deferred.")
hits = holo.search_text("gravity wells", top_k=3)

for trace, score in hits:
    print(holo.summarize_hit(trace, score))

holo.save("memory_store.json")
```

Reload later with `Hologram.load("memory_store.json", use_clip=False)`.
(This restores the gravity field state instantly without replaying history.)

---

## Demos
- Text only: `python demo.py`
- Negation-aware gravity: `python demo_negation.py`
- Reinforcement-based decay: `python demo_decay.py`
- Text → image: `python demo_clip.py`
- Image → image: `python demo_img2img.py`

Sample PNGs live in `data/`. Replace them with your own assets for better matches. CLIP demos automatically fall back to hashing when dependencies are missing.

---

## Chat CLI

```bash
python chat_cli.py --session alice --memory memory_store.json
```

- Stores each turn in holographic memory and appends JSONL logs under `chatlogs/`.
- Replays `--session-window` recent turns and pulls cross-session memories via semantic search.
- Uses `OPENAI_API_KEY` by default; falls back to an echo provider when the key or `openai` lib is missing.
- Add `--use-clip` to run the chat with CLIP encoders.

---

## FastAPI Surface

```bash
uvicorn api_server.main:app --reload
```

- `POST /add_text?glyph_id=<id>&text=<content>` → stores a trace and returns the generated `trace_id`.
- `GET /search?q=<query>&top_k=5` → returns ranked traces as `{trace, score}` dictionaries.

Both endpoints rely on the in-process `Hologram`; adjust `Hologram.init(...)` in `api_server/main.py` to toggle CLIP or gravity.

---

## Tests

```bash
pip install pytest
pytest
```

Tests cover chat session logging, the in-memory store, and JSON persistence.

---

## Repository Layout

```
hologram/            core package (API, embeddings, gravity, chatbot, demos)
api_server/          FastAPI entry point
chat_cli.py          chat demo (OpenAI or echo fallback)
data/                sample PNGs (cat.png, dog.png)
demo.py              text-only walkthrough
demo_clip.py         text→image entry point
demo_img2img.py      image→image entry point
tests/               pytest suite
```

---

## Troubleshooting
- `ModuleNotFoundError: faiss`: install `faiss-cpu` (or GPU variant) or call `Hologram.init(use_gravity=False)`.
- `RuntimeError: open_clip`: only enable CLIP (`--use-clip`, `use_clip=True`) when `torch` + `open_clip_torch` + `pillow` are installed.
- Missing sample images: add your own `cat.png` / `dog.png` (or tweak the demos to point at your data).

---

## License

MIT
