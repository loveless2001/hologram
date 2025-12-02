# Hologram Memory

A holographic memory sandbox that anchors multi-modal traces to glyphs, stores them in a lightweight vector index, and experiments with "memory gravity" fields to model concept drift.

---

## ✨ Latest Features (Nov 2023)

### 🔬 GLiNER-Powered Concept Decomposition
- **Automatic sentence → atomic concepts**: Full sentences are decomposed into semantic units using GLiNER (Generalist Named Entity Recognition)
- **Relation extraction**: Captures verbs and actions (e.g., "hit", "describes") to preserve Subject→Verb→Object flow
- **Order preservation**: Extracted concepts maintain narrative order for better memory reconstruction
- **Labels**: `concept`, `entity`, `phenomenon`, `object`, `theory`, `action`, `relationship`, `interaction`, `verb`

### 🔍 Semantic Search Interface
- **REST API**: `/search` endpoint for keyword-based semantic search
- **Streamlit UI**: Two-tab interface with Chat and Semantic Search
- **Visual results**: Color-coded similarity scores (🟢 80%+, 🟡 60-79%, 🔵 <60%)
- **Adjustable results**: Query 1-20 top matches

### 🚀 Phase 4: Field-Level Evaluation (New!)
- **Stress Testing Suite**: Validates system coherence under pressure (intra-field stability, deformation, probe drift, memory degradation).
- **Dynamic Topology**: Field deforms correctly when contradictory information is added (e.g., "Cats are solitary" vs "Cats are social").
- **Robust Similarity**: Geometric scaling ensures relation strengths are meaningful (40-95%) rather than collapsing to 100%.
- **Pollution Control**: Chat history is isolated from the concept graph to prevent system prompts from contaminating search results.

### 🧬 Concept Mitosis (Contextual Disambiguation)
- **Geometry-Based Splitting**: Uses FAISS K-means to detect bimodal distributions in concept neighbors.
- **Soft Mitosis**: Splits ambiguous concepts (e.g., "bank") into distinct variants (financial vs. river) while maintaining weak bridge links.

### 🕸️ Graph-Based Reconstruction
- **Structured Retrieval**: Instead of a flat list, retrieves a semantic subgraph (nodes + mass + relations).
- **LLM Synthesis**: Prompts the LLM with the structured graph JSON, enabling it to "reason" over the connections and synthesize a coherent narrative from memory shards.

### 🚀 Phase 4.5: Performance & Scalability (New!)
- **Semantic Embeddings**: Replaced random hashing with **MiniLM** (via `sentence-transformers`) for true semantic search.
- **Scalable Storage**: **SQLite** backend allows knowledge bases to grow beyond RAM limits.
- **Vectorized Physics**: Optimized gravity simulation using matrix operations and top-k drift.
- **Concurrent Ingestion**: Multi-threaded pipeline for fast KB construction.
- **Cached Visualization**: PCA caching reduces visualization latency to < 5ms.

### 📊 Concept Visualization
- **D3.js scatter plot**: 2D PCA projection of concept space at `/viz/viz.html`
- **Interactive**: Hover tooltips, zoom/pan, auto-refresh
- **Human-readable labels**: Shows actual text content instead of hash IDs

### 🧲 MG Scorer: Semantic Quality Metrics (New!)
- **Coherence**: Measure how tightly clustered text vectors are (0-1 scale)
- **Curvature**: Detect semantic drift and topic shifts in ordered sequences
- **Entropy**: Quantify semantic dispersion across principal components
- **Collapse Risk**: Early warning system for hallucinations and contradictions
- **Use cases**: LLM output validation, memory health monitoring, retrieval quality gates

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

### Core Package (`hologram/`)
- `api.py` – public `Hologram` API (`add_text`, `add_image_path`, search, persistence)
- `chatbot.py` – chat memory, provider abstractions, CLI orchestration helpers
- `gravity.py` – concept drift simulation and FAISS-backed `GravityField` (supports state export/import)
- `embeddings.py` – hashing encoders and CLIP wrappers (text and image)
- **`text_utils.py`** – GLiNER-based concept extraction with relation/verb detection

### API Server (`api_server/`)
- `main.py` – FastAPI service with multiple endpoints:
  - `/chat` – conversational interface
  - `/search` – semantic search for keywords
  - `/viz-data` – 2D projection data for visualization
  - `/kbs` – knowledge base management (list, upload, delete)
- `static/viz.html` – D3.js concept visualization
- `static/search.html` – standalone semantic search UI

### Command-Line Tools
- `chat_cli.py` – command-line chat demo with cross-session context
- `web_ui.py` – **Streamlit interface** with semantic search

### Demos & Scripts
- `demo.py`, `demo_clip.py`, `demo_img2img.py` – runnable examples
- `scripts/seed_relativity.py` – generates test KB with Special Relativity concepts

### Tests
- `tests/test_chatbot.py` – regression coverage for chat + persistence
- `tests/test_chaos.py` – chaos testing
- `test_reconstruction.py` – validates knowledge reconstruction from seed keywords

---

## Installation

```bash
git clone https://github.com/loveless2001/hologram.git
cd hologram
python3 -m venv .venv
source .venv/bin/activate
```

### Core dependencies

```bash
pip install numpy faiss-cpu Pillow
```

### Full feature set

For concept decomposition + API + UI:
```bash
pip install fastapi uvicorn streamlit gliner transformers sentence-transformers scikit-learn
```

For OpenAI chatbot:
```bash
pip install openai
```

For CLIP embeddings (semantic image search):
```bash
pip install torch torchvision open_clip_torch
```

Development helpers:
```bash
pip install pytest
```

*(Switch to CUDA wheels if you have a GPU.)*

---

## Quickstart

### 1. Python API

```python
from hologram import Hologram

holo = Hologram.init(encoder_mode="minilm", use_gravity=True)
holo.glyphs.create("🝞", title="Curvature Anchor")

trace_id = holo.add_text("🝞", "Memory is gravity. Collapse deferred.")
hits = holo.search_text("gravity wells", top_k=3)

for trace, score in hits:
    print(holo.summarize_hit(trace, score))

holo.save("memory_store.json")
```

Reload later with `Hologram.load("memory_store.json", use_clip=False)`.

### 2. Concept Decomposition

```python
from hologram.text_utils import extract_concepts

text = "Special Relativity describes how time dilation occurs near the speed of light."
concepts = extract_concepts(text)
# Returns: ['Special Relativity', 'time dilation', 'speed of light']
```

### 3. Streamlit UI

```bash
# Start API server
uvicorn api_server.main:app --port 8000

# In another terminal, start Streamlit UI
streamlit run web_ui.py
```

Then:
1. Select `relativity.txt` in sidebar
2. Click **🔄 Load KB**
3. Use **🔍 Semantic Search** tab to search for keywords
4. Or use **💬 Chat** tab for conversational queries

### 4. Visualization

Visit `http://localhost:8000/viz/viz.html` after loading a KB to see the 2D concept projection.

---

## API Endpoints

### Knowledge Base Management
- `GET /kbs` – list available knowledge bases
- `POST /kbs/upload` – upload new KB (text file)
- `DELETE /kbs/{name}` – delete KB

### Memory Operations
- `POST /chat` – conversational interface (loads KB if `kb_name` provided)
  ```json
  {"message": "What is time dilation?", "kb_name": "relativity.txt"}
  ```

- `POST /search` – semantic search
  ```json
  {"query": "speed of light", "top_k": 10}
  ```
  Returns:
  ```json
  {
    "query": "speed of light",
    "results": [
      {"content": "speed of light", "score": 1.0},
      {"content": "spacetime", "score": 0.74}
    ]
  }
  ```

### Visualization
- `GET /viz-data` – returns 2D projection points and labels
  ```json
  {
    "points": [[x1, y1], [x2, y2], ...],
    "labels": ["concept1", "concept2", ...]
  }
  ```

---

## Knowledge Base Format

### Text Files (`.txt`)
Place in `data/kbs/`. Each line is processed as follows:
1. **GLiNER extraction**: Sentence → atomic concepts
2. **Memory storage**: Each concept added to holographic memory
3. **Gravity field**: Concepts positioned in vector space

Example (`data/kbs/relativity.txt`):
```
Special Relativity describes how time dilation occurs near the speed of light.
Length contraction is observed when an object moves at high velocity.
Mass-energy equivalence states that energy equals mass times the speed of light squared.
```

### JSON Files (`.json`)
Full memory snapshots with gravity state. Generated via:
```python
memory.save("relativity_kb.json")
```

---

## Demos

- Text only: `python demos/demo.py`
- Negation-aware gravity: `python demos/demo_negation.py`
- Reinforcement-based decay: `python demos/demo_decay.py`
- Text → image: `python demos/demo_clip.py`
- Image → image: `python demos/demo_img2img.py`
- **Knowledge reconstruction**: `python tests/test_reconstruction.py`

---

## Chat CLI

```bash
python chat_cli.py --session alice --memory memory_store.json
```

- Stores each turn in holographic memory and appends JSONL logs under `chatlogs/`
- Replays `--session-window` recent turns and pulls cross-session memories via semantic search
- Uses `OPENAI_API_KEY` by default; falls back to an echo provider when the key or `openai` lib is missing
- Add `--use-clip` to run the chat with CLIP encoders

---

## Tests

```bash
pip install pytest
pytest
```

Tests cover:
- Chat session logging
- In-memory store operations
- JSON persistence
- Chaos testing for visualization

---

## Repository Layout

```
hologram/            Core package
  ├── api.py         Hologram API (add_text, search, persistence)
  ├── chatbot.py     Chat memory and provider abstractions
  ├── gravity.py     Gravity field simulation (concept drift)
  ├── embeddings.py  Text/image encoders (hash + CLIP)
  └── text_utils.py  GLiNER-based concept extraction

api_server/          FastAPI service
  ├── main.py        REST API (/search, /chat, /viz-data, /kbs)
  └── static/        
      ├── viz.html   D3.js concept visualization
      └── search.html Standalone search interface

demos/               Demonstration scripts
  ├── demo.py        Text-only walkthrough
  ├── demo_clip.py   Text → image search
  ├── demo_img2img.py Image → image similarity
  ├── demo_negation.py Negation-aware gravity
  ├── demo_decay.py  Reinforcement-based decay
  ├── demo_knowledge_base.py KB construction
  └── demo_kg_comparison.py KG comparison

tests/               Test suite
  ├── test_chatbot.py Chat memory tests (pytest)
  ├── test_chaos.py  Chaos testing
  ├── test_api.py    API endpoint validation
  ├── test_gliner.py GLiNER extraction tests
  ├── test_reconstruction.py Knowledge reconstruction
  ├── test_search_relations.py Relation extraction tests
  └── benchmark.py   Performance benchmarks

scripts/             Utility scripts
  └── seed_relativity.py Generate test KB

data/                Data files
  ├── kbs/           Knowledge base text files
  ├── cat.png        Sample images (for CLIP demos)
  └── dog.png

docs/                Documentation

Root scripts:
  ├── chat_cli.py    Command-line chat interface
  ├── web_ui.py      Streamlit UI (chat + search)
  └── run_ui.sh      Quick launch script
```

---

## How It Works

### Concept Decomposition Pipeline
1. **Input**: Full sentence (e.g., "Time dilation occurs near the speed of light")
2. **GLiNER**: Extracts entities with labels `[concept, entity, phenomenon, action, ...]`
3. **Order Preservation**: Sorts by appearance to maintain Subject→Verb→Object flow
4. **Output**: `['Time dilation', 'occurs', 'speed of light']`

### Knowledge Reconstruction (SMI)
1. **Seed keyword**: User provides (e.g., "speed of light")
2. **Probe Simulation**: A probe drifts through the gravity field, attracted by massive concepts.
3. **Packet Extraction**: The system captures the "local field" around the probe's trajectory.
4. **LLM Synthesis**: The LLM receives the **Memory Packet** (SMI) and reconstructs the narrative.

Example SMI Packet:
```json
{
  "seed": "speed of light",
  "nodes": [
    {"name": "time dilation", "mass": 1.3, "age": 12},
    {"name": "speed of light", "mass": 1.5, "age": 4}
  ],
  "edges": [
    {"a": "speed of light", "b": "time dilation", "relation": 0.82, "tension": 0.12}
  ],
  "glyphs": [
    {"id": "🝞", "mass": 2.4}
  ]
}
```
LLM Output: *"The speed of light is intrinsically linked to time dilation, a relationship anchored by the curvature glyph (🝞)..."*

### Contextual Disambiguation (Mitosis)
1. **Detection**: System monitors concepts for "semantic tension" (distinct neighbor clusters).
2. **Trigger**: If clusters diverge beyond a threshold, **Mitosis** occurs.
3. **Split**: `Field` -> `Field_1` (Physics context), `Field_2` (Agri context).
4. **Bridge**: A weak link connects them, allowing "wormhole" jumps for creative associations.

---

## Troubleshooting

### Installation Issues
- `ModuleNotFoundError: faiss`: install `faiss-cpu` (or GPU variant) or call `Hologram.init(use_gravity=False)`
- `RuntimeError: open_clip`: only enable CLIP (`--use-clip`, `use_clip=True`) when `torch` + `open_clip_torch` + `pillow` are installed
- Missing sample images: add your own `cat.png` / `dog.png` (or tweak the demos to point at your data)

### GLiNER Issues
- **Slow loading**: Model downloads on first use (~600MB). Cached for subsequent runs.
- **KeyError during extraction**: Ensure labels list is deduplicated (handled in `text_utils.py`)
- **Missing verbs**: Some abstract verbs may not be captured. Adjust threshold (default: 0.25) in `extract_concepts()`

### API Server
- **404 on /search**: Restart server after adding endpoint
- **"No KB loaded" error**: Use `/chat` with `{"message": "load", "kb_name": "relativity.txt"}` first
- **Empty visualization**: Ensure KB is loaded and contains processed concepts

---

## Performance Notes

- **GLiNER model**: ~600MB download, CPU-efficient for inference
- **Vector index**: FAISS with GPU acceleration if available
- **Concept extraction**: ~0.5-2s per sentence on CPU (cached model)
- **Search latency**: < 1ms (FAISS + MiniLM)
- **Visualization**: < 5ms (PCA Cached)

---

## License

MIT
