# Hologram Memory

A holographic memory sandbox that anchors multi-modal traces to glyphs, stores them in a lightweight vector index, and experiments with "memory gravity" fields to model concept drift.

---

## ✨ Latest Features (Dec 2024)

### 🔬 Glyph-Conditioned Spectral Memory (Apr 2026)
- **Glyph-Routed Retrieval**: Queries route through glyph-conditioned subspaces instead of one global index
- **GlyphOperator**: Per-glyph transform `T_g(z) = P_k R_g z` with orthogonal rotation + dimension projection
- **GlyphRouter**: Infers glyph distribution from query, searches per-glyph FAISS shard indexes, merges results
- **Discriminant Basis**: Centroid-based discriminant projection finds directions that maximally separate glyphs
- **Benchmark Result**: +14% recall, 90% interference reduction vs global cosine on MiniLM real-text embeddings
- **API**: `search_routed()` Python method + `POST /query/routed` REST endpoint
- Modules: `hologram/glyph_operator.py`, `hologram/glyph_router.py`

### 🧭 KG + Drift Pivot Notes (Mar 2026)
- Added initial `hologram/kg/` module for batch semantic graph snapshots.
- Added initial `hologram/drift/` module for flexible batch-vs-batch drift scoring.
- Added API endpoints:
  - `POST /kg/build_batch`
  - `POST /drift/compare`
- Resume reference: `docs/kg_drift_resume.md`

### 💻 Code Mapping Layer (New!)
- **AST-Based Parsing**: Extracts classes, functions, and symbols from Python source files
- **Precise Source Mapping**: Maps each concept to exact file path and line span
- **Semantic Code Search**: Query code using natural language ("authentication logic", "orbit math")
- **Fusion Protection**: Code concepts maintain file-specific identity, preventing incorrect merging
- **Docstring Integration**: Attaches documentation as traces for richer semantic understanding
- **REST API**: `/ingest/code` and `/query/code` endpoints for programmatic access

### 🧹 Spelling Correction & Normalization Pipeline
- **4-Stage Pipeline**: Cleans noisy input before it enters the semantic field
  - **Stage 0**: Tokenization (unicode cleanup, space normalization)
  - **Stage 1**: SymSpell dictionary-backed spell correction with gravity whitelist protection
  - **Stage 2**: Optional LLM-based contextual rewrite (disabled by default)
  - **Stage 3**: Manifold alignment (semantic near-neighbor mapping via gravity field)
  - **Stage 4**: Canonicalization (lowercase, hyphen/underscore removal)
- **Prevents Pollution**: Stops "gravty" and "gravity" from creating duplicate concepts
- **Whitelist Protection**: Won't "correct" domain-specific terms like "QFT" or "SM"
- **Read-Only Gravity**: Uses field for reference without modifying it during normalization
- **Runs First**: Executes before coreference resolution for maximum effectiveness

### 🔗 Hybrid Coreference Resolution
- **FastCoref Integration**: High-accuracy pronoun resolution using neural coreference models
- **Gravity Fallback**: Vector-based resolution for ambiguous deictics ("this", "that") using concept mass attraction
- **Trace Metadata**: Stores both original and resolved text with pronoun→antecedent mappings
- **Improved Extraction**: GLiNER sees resolved text, reducing concept fragmentation
- **Configuration**: Toggle via `Config.coref.ENABLE_COREF` and `Config.coref.ENABLE_GRAVITY_FALLBACK`

### 🔬 GLiNER-Powered Concept Decomposition
- **Automatic sentence → atomic concepts**: Full sentences are decomposed into semantic units using GLiNER (Generalist Named Entity Recognition)
- **Relation extraction**: Captures verbs and actions (e.g., "hit", "describes") to preserve Subject→Verb→Object flow
- **Order preservation**: Extracted concepts maintain narrative order for better memory reconstruction
- **Labels**: `concept`, `entity`, `phenomenon`, `object`, `theory`, `action`, `relationship`, `interaction`, `verb`

### 🔍 Project-Based Memory API
- **REST API**: Project-scoped ingestion and query endpoints
- **Probe Physics**: Dynamic retrieval using gravitational field simulation
- **Tier-Aware**: Supports 3-tier ontology (Domain/System/Meta)
- **Persistence**: Auto-save to `~/.hologram_memory/<project>/`

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

### 🧲 MG Scorer: Semantic Quality Metrics
- **Coherence**: Measure how tightly clustered text vectors are (0-1 scale)
- **Curvature**: Detect semantic drift and topic shifts in ordered sequences
- **Entropy**: Quantify semantic dispersion across principal components
- **Collapse Risk**: Early warning system for hallucinations and contradictions
- **Use cases**: LLM output validation, memory health monitoring, retrieval quality gates

### 💰 Cost Engine: Diagnostic Meta-Layer (New!)
- **Resistance**: Measures integration difficulty for new concepts
- **Entropy**: Quantifies neighborhood disorder (cognitive load)
- **Drift Cost**: Tracks semantic field instability over time
- **Suggestions**: Auto-recommends split/fuse/stabilize actions
- **Presets**: Analytical, creative, conservative configurations

### ⚙️ Global Configuration System (New!)
- **Persistent storage**: Configuration saved to `~/.hologram_memory/_hologram_system/memory.db`
- **Single source**: `hologram/config.py` for all system parameters
- **Environment overrides**: `HOLOGRAM_USE_GPU`, `HOLOGRAM_PORT`, `HOLOGRAM_MEMORY_DIR`
- **Auto-sync**: Server loads global config on startup, env vars still override
- **CLI tool**: `scripts/setup_hologram.py` for initialization and conflict resolution
- **Config hierarchy**: `Environment Variables > Global Config > Defaults`
- **Interactive setup**: Prompts for conflict resolution when local and global differ
- **System-managed**: Global project marked as read-only for safety

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
- `api.py` – public `Hologram` API (`add_text`, `add_image_path`, `ingest_code`, search, persistence)
- `server.py` – **FastAPI server** with REST endpoints for VSCode extension
- `chatbot.py` – chat memory, provider abstractions, CLI orchestration helpers
- `gravity.py` – concept drift simulation and FAISS-backed `GravityField`
- `embeddings.py` – MiniLM, CLIP, and hashing encoders
- `text_utils.py` – GLiNER-based concept extraction
- `normalization.py` – **4-stage spelling correction & normalization pipeline**
- `coref.py` – **hybrid coreference resolution**
- `config.py` – **centralized configuration**
- `global_config.py` – **global config persistence & sync**
- `cost_engine.py` – **diagnostic metrics**
- `manifold.py` – vector space alignment
- `glyph_operator.py` – **glyph-conditioned transform operators** (rotation + projection)
- `glyph_router.py` – **glyph-routed retrieval** (shard routing, discriminant basis, score fusion)
- `retrieval.py` – probe-based dynamic retrieval
- `smi.py` – Symbolic Memory Interface
- `mg_scorer.py` – semantic quality metrics
- `code_map/` – **Code Mapping Layer** (NEW)
  - `parser.py` – AST-based Python code parsing
  - `extractor.py` – Symbol extraction and normalization
  - `mapper.py` – Concept-to-Glyph mapping with source metadata
- `storage/` – SQLite backend for scalable persistence
- `kg/` – batch knowledge graph snapshot builder (pivot scaffolding)
- `drift/` – drift detectors and report engine (pivot scaffolding)

### Command-Line Tools
- `chat_cli.py` – command-line chat demo with cross-session context
- `web_ui.py` – **Streamlit interface** with semantic search

### Demos & Scripts
- `demo.py`, `demo_clip.py`, `demo_img2img.py` – runnable examples
- `scripts/seed_relativity.py` – generates test KB with Special Relativity concepts
- `scripts/setup_hologram.py` – **global configuration management** (NEW)

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

For concept decomposition + coreference + API + UI:
```bash
pip install fastapi uvicorn streamlit gliner transformers sentence-transformers scikit-learn fastcoref
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

### 3. Coreference Resolution

```python
from hologram.coref import resolve

text = "The fusion engine is unstable. It requires cooling."
resolved_text, coref_map = resolve(text)

print(resolved_text)
# "The fusion engine is unstable. The fusion engine requires cooling."

print(coref_map)
# {'It': 'The fusion engine'}
```

**Automatic Integration**: When using `Hologram.add_text()`, coreference resolution runs automatically if enabled in config.

### 4. Start Server

```bash
# Start Hologram server (default: localhost:8000)
python -m hologram.server

# Or with custom settings
HOLOGRAM_PORT=9000 HOLOGRAM_USE_GPU=0 python -m hologram.server
```

Then:
1. Use REST API at `http://localhost:8000`
2. Or start Streamlit UI: `streamlit run web_ui.py`

### 5. Configuration Management

```bash
# Initialize global configuration (first-time setup)
python scripts/setup_hologram.py --init

# View current configuration
python scripts/setup_hologram.py --show    # Local config
python scripts/setup_hologram.py --global  # Global config

# Auto-sync (for scripts/automation)
python scripts/setup_hologram.py --sync

# Interactive mode (prompts for conflict resolution)
python scripts/setup_hologram.py
```

**Configuration Priority:**
```
Environment Variables > Global Config > Defaults
```

**Example: Override settings**
```bash
# Temporary override (this session only)
HOLOGRAM_PORT=9000 python -m hologram.server

# Permanent override (update global config)
python scripts/setup_hologram.py --init  # Updates global DB
```


### 6. Glyph-Routed Retrieval

```python
from hologram import Hologram

holo = Hologram.init(encoder_mode="minilm", use_gravity=True)

# Create domain-specific glyphs
holo.glyphs.create("physics", title="Physics")
holo.glyphs.create("biology", title="Biology")

# Add traces to glyphs
holo.add_text("physics", "General relativity describes gravity as spacetime curvature")
holo.add_text("biology", "DNA replication copies genetic material in cells")

# Glyph-routed search (routes through glyph subspaces)
results = holo.search_routed("How does gravity work?", top_k=3)
for trace, score in results:
    print(f"{score:.3f}: {trace.content}")
```

### 7. Code Ingestion

```python
from hologram import Hologram

holo = Hologram.init(encoder_mode="minilm", use_gravity=True)

# Ingest a Python source file
num_concepts = holo.ingest_code("/path/to/source.py")
print(f"Extracted {num_concepts} code concepts")

# Query for code
results = holo.query_code("authentication logic", top_k=5)
for result in results:
    print(f"{result['concept']} in {result['file']} at lines {result['span']}")
```

### 7. Visualization

Visit `http://localhost:8000/viz/viz.html` after loading a KB to see the 2D concept projection.

---

## API Endpoints

### Core Operations
- `GET /` – health check and server status
- `POST /ingest` – ingest text into project memory
- `POST /ingest/code` – ingest source code file (extracts symbols, maps to concepts)
  ```json
  {
    "project": "my_project",
    "path": "/path/to/source.py",
    "tier": 1
  }
  ```
  ```json
  {
    "project": "my_project",
    "text": "The fusion engine uses magnetic confinement.",
    "origin": "code",  // code, docs, config, manual
    "tier": 1,  // 1=Domain, 2=System
    "metadata": {"file": "engine.py", "line": 42}
  }
  ```

- `POST /query` – semantic query using probe physics
  ```json
  {
    "project": "my_project",
    "text": "How does the engine work?",
    "top_k": 5
  }
  ```
  Returns:
  ```json
  {
    "query": "How does the engine work?",
    "nodes": [{"name": "fusion engine", "mass": 2.3, ...}],
    "edges": [{"a": "engine", "b": "magnetic", "relation": 0.82}],
    "glyphs": [{"id": "file:engine.py", "mass": 5.1}],
    "trajectory_steps": [[0.1, 0.2, ...], ...]
  }
  ```

- `POST /query/routed` – glyph-routed retrieval (queries route through glyph subspaces)
  ```json
  {
    "project": "my_project",
    "text": "How does gravity bend spacetime?",
    "top_k": 5
  }
  ```
  Returns:
  ```json
  {
    "query": "How does gravity bend spacetime?",
    "results": [
      {"trace_id": "physics_0", "content": "Einstein's theory...", "score": 0.89}
    ]
  }
  ```

- `POST /query/code` – query specifically for code concepts
  ```json
  {
    "project": "my_project",
    "text": "authentication logic",
    "top_k": 5
  }
  ```
  Returns:
  ```json
  {
    "results": [
      {
        "concept": "function:authenticate_user",
        "file": "/path/to/auth.py",
        "span": [42, 67],
        "score": 0.89,
        "snippet": "def authenticate_user(username, password)..."
      }
    ]
  }
  ```

### Project Management
- `GET /projects` – list all active projects
- `GET /memory/{project}` – get memory summary (concept counts, recent traces)
- `POST /save/{project}` – save project to disk
- `POST /load/{project}` – load project from disk
- `DELETE /project/{project}` – remove project from memory

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
  ├── server.py      FastAPI server (REST endpoints)
  ├── chatbot.py     Chat memory and provider abstractions
  ├── gravity.py     Gravity field simulation (concept drift)
  ├── embeddings.py  Text/image encoders (MiniLM, CLIP, hash)
  ├── text_utils.py  GLiNER-based concept extraction
  ├── normalization.py Spelling correction & normalization pipeline (NEW)
  ├── coref.py       Hybrid coreference resolution
  ├── config.py      Centralized configuration
  ├── cost_engine.py Diagnostic metrics
  ├── manifold.py    Vector space alignment
  ├── retrieval.py   Probe-based retrieval
  ├── smi.py         Symbolic Memory Interface
  ├── mg_scorer.py   Semantic quality metrics
  └── storage/       SQLite backend

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
  ├── test_coref.py  Coreference resolution tests (NEW)
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

### Ingestion Pipeline
1. **Input**: Full sentence (e.g., "The gravty feild failed. It was unstable.")
2. **Normalization Pipeline** (NEW):
   - **SymSpell**: Corrects "gravty feild" → "gravity field"
   - **Whitelist Check**: Preserves domain terms (e.g., "QFT", "SM")
   - **Manifold Alignment**: Maps near-miss tokens to existing concepts
   - **Canonicalization**: Standardizes formatting
   - **Output**: "The gravity field failed. It was unstable."
3. **Coreference Resolution**: 
   - **FastCoref**: Resolves "It" → "The gravity field"
   - **Gravity Fallback**: For ambiguous deictics, finds nearest concept by mass-weighted similarity
   - **Output**: "The gravity field failed. The gravity field was unstable."
4. **GLiNER Extraction**: Extracts entities from resolved text
   - Labels: `[concept, entity, phenomenon, action, ...]`
   - Order preserved: Subject→Verb→Object flow
5. **Output**: `['gravity field', 'failed', 'unstable']` (with correct antecedents and clean spelling)

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
- **Server not responding**: Check if port 8000 is already in use
- **Project not found**: Use `/ingest` to create a project first
- **Empty query results**: Ensure project has ingested data

---

## Performance Notes

- **GLiNER model**: ~600MB download, CPU-efficient for inference
- **Vector index**: FAISS with GPU acceleration if available
- **Concept extraction**: ~0.5-2s per sentence on CPU (cached model)
- **Search latency**: < 1ms (FAISS + MiniLM)
- **Visualization**: < 5ms (PCA Cached)
- **Storage**: SQLite backend (default) for scalable persistence, JSON also supported

---

## License

MIT
