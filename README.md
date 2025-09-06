# Hologram Memory

A minimal **holographic memory starter kit** that demonstrates how to anchor multi-modal traces (text + images) to **glyphs** and retrieve them with semantic search.  
Built with a tiny in-memory store + OpenCLIP for text ↔ image alignment.

---

## Features
- **Glyph Anchors** → symbolic IDs (🝞, 🜂, `memory:gravity`, etc.) that link traces.
- **Text & Image Embeddings**
  - Hash-based encoders for lightweight demos.
  - [OpenCLIP](https://github.com/mlfoundations/open_clip) for real text–image semantic search.
- **Memory Store**
  - In-memory vector index with cosine similarity.
  - Drop-in replacement for FAISS/ScaNN if scaling.
- **APIs**
  - `add_text()`, `add_image_path()`, `recall_glyph()`, `search_text()`, `search_image_path()`.
- **Demos**
  - `demo.py` → text only.
  - `demo_clip.py` → text → image retrieval.
  - `demo_img2img.py` → image → image similarity.

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

Put images in `./data/` (e.g. `cat.jpg`, `dog.jpg`):

```bash
python demo_clip.py
```

Expected: `"a photo of a cat"` ranks `cat.jpg` above `dog.jpg`.

### Image → Image demo

```bash
python demo_img2img.py
```

Expected: `cat2.jpg` is closer to `cat1.jpg` than `dog1.jpg`.

---

## Repo Layout

```
hologram/         # core package
  config.py       # constants
  embeddings.py   # hashing + CLIP encoders
  store.py        # memory store + vector index
  glyphs.py       # glyph registry
  api.py          # main Hologram API
demo.py           # text-only demo
demo_clip.py      # text→image demo
demo_img2img.py   # image→image demo
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

👉 Want me to also generate a **`requirements.txt`** so others can `pip install -r requirements.txt` instead of manually typing packages?
```
