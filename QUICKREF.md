# Hologram System - Quick Reference Guide

## 🚀 Getting Started (30 seconds)

```bash
# 1. Start API Server
uvicorn api_server.main:app --port 8000

# 2. Start Streamlit UI (in another terminal)
streamlit run web_ui.py

# 3. In browser:
#    - Select "relativity.txt" 
#    - Click "🔄 Load KB"
#    - Switch to "🔍 Semantic Search" tab
#    - Search for "speed of light"
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
- [x] **GLiNER-based decomposition** (sentences → atomic concepts)
- [x] **Relation extraction** (verbs, actions, interactions)
- [x] **Order preservation** (maintains Subject→Verb→Object flow)
- [x] **Configurable labels** for entity types

#### API Endpoints
- [x] `/chat` - Conversational interface
- [x] `/search` - Semantic keyword search
- [x] `/viz-data` - 2D projection data
- [x] `/kbs` - Knowledge base management (list/upload/delete)

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

#### Phase 4: Dynamic Graph Retrieval (New!)
- [x] **Probe Dynamics**: Physics-aware retrieval trajectory
- [x] **SMI**: Structured Memory Packet for LLM
- [x] **Hallucination Control**: LLM restricted to field data

---

## 🎯 Use Cases

### 1. Knowledge Base Q&A
**Goal**: Search domain knowledge with keywords  
**How**: Use Streamlit search tab or `/search` endpoint  
**Example**: "speed of light" → Returns related physics concepts

### 2. Concept Exploration
**Goal**: Visualize how concepts cluster in semantic space  
**How**: Visit `/viz/viz.html` after loading KB  
**Example**: See "time dilation" near "spacetime" and "Special Relativity"

### 3. Memory Reconstruction
**Goal**: Reconstruct context from partial information  
**How**: Run `test_reconstruction.py` or use Chat  
**Example**: From "gravity" → retrieves graph (mass, relations) → LLM explains concept
**Feature**: Uses **Graph-Based Reconstruction** for higher coherence

### 4. Conversational AI
**Goal**: Chat with knowledge base  
**How**: Use Streamlit chat tab or `chat_cli.py`  
**Example**: "What is time dilation?" → LLM response with memory context

---

## 🔧 Common Workflows

### Workflow A: Add New Knowledge Base

```bash
# 1. Create text file
echo "Quantum entanglement connects distant particles.
Superposition allows particles to exist in multiple states.
Wave function collapse determines measurement outcomes." > data/kbs/quantum.txt

# 2. Load in Streamlit UI
#    - Upload via sidebar OR select from list
#    - Click "🔄 Load KB"

# 3. Verify
#    - Search for "quantum" in search tab
#    - Check visualization at /viz/viz.html
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

### Workflow C: Semantic Search via API

```bash
# Load KB
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "load", "kb_name": "relativity.txt"}'

# Search
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "time", "top_k": 5}' | jq
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
- **Single KB at a time**: API server loads one KB in memory
- **No multi-tenancy**: Concurrent users share same KB state
- **No persistence auto-save**: Changes lost on server restart (unless explicitly saved)

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
# Check GLiNER model cache
ls ~/.cache/huggingface/hub/ | grep gliner

# Test search endpoint directly
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "top_k": 1}'

# Run reconstruction test
python test_reconstruction.py

# Check concept count
curl -s "http://localhost:8000/viz-data" | jq '.points | length'
```

---

**Last Updated**: 2025-11-27  
**System Version**: Hologram v1.0 (Phase 4: Dynamic Graph Retrieval)
