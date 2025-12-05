# Hologram System - Quick Reference Guide

## 🚀 Getting Started (30 seconds)

```bash
# 1. Start Hologram Server
python -m hologram.server

# 2. Test the API
curl http://localhost:8000/

# 3. Ingest some text
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"project": "demo", "text": "Hologram is a semantic memory system.", "origin": "manual"}'

# 4. Query the memory
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project": "demo", "text": "memory", "top_k": 3}'
```

---

## 📋 Feature Checklist

### ✅ Implemented Features

#### Core System
- [x] **Holographic memory** with gravity field simulation
- [x] **Vector embeddings** (hash-based + optional CLIP)
- [x] **FAISS indexing** for efficient semantic search
- [x] **JSON persistence** with gravity state

#### Concept Processing
- [x] **Coreference Resolution** (hybrid FastCoref + Gravity fallback)
- [x] **GLiNER-based decomposition** (sentences → atomic concepts)
- [x] **Relation extraction** (verbs, actions, interactions)
- [x] **Order preservation** (maintains Subject→Verb→Object flow)
- [x] **Configurable labels** for entity types

#### API Endpoints
- [x] `/` - Health check
- [x] `/ingest` - Ingest text into project memory
- [x] `/query` - Probe-based semantic query
- [x] `/memory/{project}` - Get memory summary
- [x] `/save/{project}` - Save project memory
- [x] `/load/{project}` - Load project memory
- [x] `/projects` - List active projects
- [x] `/project/{project}` - Delete project (DELETE)

#### User Interfaces
- [x] **Streamlit UI** with:
  - Chat tab
  - Semantic search tab
  - KB management sidebar
- [x] **D3.js visualization** (interactive concept map)
- [x] **Standalone search HTML** (optional interface)

#### Knowledge Reconstruction
- [x] Seed keyword → related concepts
- [x] Semantic clustering via vector similarity
- [x] **Graph-based synthesis** (LLM reconstructs from structured subgraph)
- [x] Validation test scripts

#### Contextual Intelligence
- [x] **Concept Mitosis**: Auto-splits ambiguous terms (e.g., "Field") based on context
- [x] **Bridge Links**: Preserves metaphorical connections between split concepts
- [x] **Dynamic Graph Evolution**: Memory adapts to new domains automatically

#### Phase 4: Dynamic Graph Retrieval
- [x] **Probe Dynamics**: Physics-aware retrieval trajectory
- [x] **SMI**: Structured Memory Packet for LLM
- [x] **Hallucination Control**: LLM restricted to field data

#### Cost Engine (New!)
- [x] **Diagnostic Metrics**: Resistance, Entropy, Drift Cost
- [x] **Suggestions**: Split, fuse, stabilize, no-action
- [x] **Configurable Presets**: Analytical, creative, conservative

#### Configuration System (New!)
- [x] **Centralized Config**: `hologram/config.py`
- [x] **Environment Overrides**: `HOLOGRAM_USE_GPU`, `HOLOGRAM_PORT`, etc.
- [x] **Optimal Defaults**: Auto-detects GPU, uses MiniLM embeddings

---

## 🎯 Use Cases

### 1. Code Memory (VSCode Extension)
**Goal**: Track semantic relationships across codebase  
**How**: Use `/ingest` endpoint to add code snippets  
**Example**: Ingest function definitions → Query for "authentication logic"

### 2. Project Knowledge Base
**Goal**: Build domain-specific memory per project  
**How**: Use project-scoped ingestion and queries  
**Example**: Ingest docs → Query for implementation details

### 3. Semantic Query
**Goal**: Find related concepts using probe physics  
**How**: Use `/query` endpoint with natural language  
**Example**: "How does authentication work?" → Returns relevant code/docs nodes

### 4. Memory Persistence
**Goal**: Save and restore project memory across sessions  
**How**: Use `/save/{project}` and `/load/{project}` endpoints  
**Example**: Save at end of day → Load next morning

---

## 🔧 Common Workflows

### Workflow A: Create and Query a Project

```bash
# 1. Ingest text into project
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "physics",
    "text": "Quantum entanglement connects distant particles.",
    "origin": "docs",
    "tier": 1
  }'

# 2. Add more concepts
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "physics",
    "text": "Superposition allows particles to exist in multiple states.",
    "origin": "docs",
    "tier": 1
  }'

# 3. Query the memory
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"project": "physics", "text": "quantum", "top_k": 5}' | jq

# 4. Save the project
curl -X POST "http://localhost:8000/save/physics"
```

### Workflow B: Test Concept Extraction

```bash
# Quick test
source .venv/bin/activate
python3 -c "
from hologram.text_utils import extract_concepts
print(extract_concepts('Einstein developed the theory of relativity.'))
"
# Output: ['Einstein', 'theory of relativity', ...]
```

### Workflow C: Ingest and Query via API

```bash
# Ingest text
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "my_project",
    "text": "Einstein developed the theory of relativity.",
    "origin": "docs",
    "tier": 1
  }'

# Query memory
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "project": "my_project",
    "text": "relativity",
    "top_k": 5
  }' | jq

# Get memory summary
curl "http://localhost:8000/memory/my_project" | jq
```

---

## 📊 Current System State

### Models & Dependencies
- **GLiNER**: `urchade/gliner_medium-v2.1` (~600MB)
- **FAISS**: CPU/GPU vector index
- **Embeddings**: Hash-based (512-dim) OR CLIP (optional)

### Knowledge Bases
Location: `data/kbs/`
- `relativity.txt` - Special Relativity concepts (demo KB)
- Custom KBs can be added as `.txt` files

### Extracted Concepts
Currently stored: ~20 concepts from relativity.txt
- Atomic: "speed of light", "time dilation", "mass", "energy"
- Relational: "hit" (from baseball example)
- Order preserved for reconstruction

### Performance Metrics
- **Concept extraction**: 0.5-2s per sentence (CPU)
- **Search latency**: <100ms for top-10 results
- **Visualization**: Real-time (refreshes every 5s)

---

## 🐛 Known Limitations

### Concept Extraction
- **Abstract verbs**: May miss verbs like "occurs", "states" in complex sentences
- **Threshold sensitivity**: Default 0.25 may need tuning per domain
- **Context loss**: Atomic concepts lose some original sentence structure

### Memory Reconstruction
- **Partial relations**: Subject→Verb→Object not always complete
- **Semantic gaps**: Related concepts depend on embedding quality
- **Order approximation**: PCA projection may distort true semantic distances

### System
- **Project-based isolation**: Each project has separate memory instance
- **SQLite storage**: Default backend (auto-migrates from legacy JSON)
- **Auto-save on shutdown**: Projects saved to `~/.hologram_memory/<project>/memory.db`
- **Tier-aware ingestion**: Supports 3-tier ontology (Domain/System/Meta)

---

## 🔮 Future Enhancements (Not Implemented)

### Short-term
- [ ] Improve verb extraction (custom relation model?)
- [ ] Add context window for reconstruction (return sentence fragments)
- [ ] Multi-KB support (switch without reload)
- [ ] Auto-save on KB modifications

### Medium-term
- [ ] Graph-based relation storage (complement vector space)
- [ ] Temporal decay for concepts (recency weighting)
- [ ] User feedback loop (reinforce/suppress concepts)
- [ ] Export reconstructed knowledge as text

### Long-term
- [ ] Multi-modal concept extraction (images + text)
- [ ] Hierarchical concept organization (ontology learning)
- [ ] Distributed memory across multiple agents
- [ ] Real-time collaboration on shared KBs

---

## 📖 Documentation Index

- **Main README**: `/README.md` - Full system documentation
- **This file**: `/QUICKREF.md` - Quick reference guide
- **API Examples**: See `/README.md` → "API Endpoints"
- **Code**: See inline docstrings in `hologram/*.py`

---

## 🆘 Getting Help

### Troubleshooting Steps
1. **Check server logs**: Look for errors in uvicorn output
2. **Verify KB loaded**: Use search tab, check for "No KB loaded" error
3. **Test endpoint**: `curl http://localhost:8000/` should return `{"status":"ok"}`
4. **Inspect visualization**: Empty viz? → KB not loaded or no concepts extracted

### Debug Commands
```bash
# Check server health
curl http://localhost:8000/ | jq

# List active projects
curl http://localhost:8000/projects | jq

# Get project memory summary
curl http://localhost:8000/memory/my_project | jq

# Test ingest endpoint
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"project": "test", "text": "Hello world", "origin": "manual"}'
```

---

**Last Updated**: 2025-12-05  
**System Version**: Hologram v1.4 (Coreference Resolution)
