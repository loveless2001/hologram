# Unused Files Analysis Report
**Generated:** 2025-12-03  
**Project:** Hologram

## Executive Summary

This report identifies potentially unused or underutilized files in the Hologram codebase based on import analysis and architectural review.

---

## 🔴 HIGH PRIORITY - Likely Unused Files

### 1. `hologram/run_sim.py` (UNUSED)
- **Status:** ❌ **No imports found**
- **Purpose:** Standalone gravity simulation script
- **Details:** 
  - Creates a `GravitySim` instance and runs a demo
  - Outputs to hardcoded paths: `/mnt/data/gravity_space.png` and `/mnt/data/gravity_space.json`
  - Not imported by any other module
  - Appears to be an early prototype/demo
- **Recommendation:** 
  - **DELETE** or move to `demos/` or `scripts/` if still useful
  - Update paths if keeping it

### 2. `hologram/demo_clip.py` (MINIMAL USE)
- **Status:** ⚠️ **Only used by `demos/demo_clip.py`**
- **Purpose:** Text-to-image retrieval demo using CLIP
- **Details:**
  - Only imported by the wrapper script in `demos/demo_clip.py`
  - The wrapper is just 4 lines that import and call `main()`
- **Recommendation:**
  - **CONSOLIDATE:** Move the code from `hologram/demo_clip.py` directly into `demos/demo_clip.py`
  - Delete `hologram/demo_clip.py` (it doesn't belong in the core library)

---

## 🟡 MEDIUM PRIORITY - Underutilized Core Modules

### 3. `hologram/config.py` (INDIRECT USE)
- **Status:** ⚠️ **No direct imports, but exports `VECTOR_DIM` and `SEED`**
- **Usage:** 
  - Imported by `embeddings.py` and `api.py` for constants
  - Only contains 2 constants:
    ```python
    VECTOR_DIM = 384
    SEED = 42
    ```
- **Recommendation:**
  - **KEEP** - This is a valid config pattern
  - Consider expanding with more configuration options or merging into a larger config system

### 4. `hologram/manifold.py` (MINIMAL USE)
- **Status:** ⚠️ **Only used by `api.py` and tested in `test_manifold.py`**
- **Purpose:** Latent manifold for vector normalization and alignment
- **Details:**
  - Core class `LatentManifold` is used in `Hologram.init()` and `Hologram.load()`
  - Provides vector normalization and encoding alignment
  - Only 44 lines of code
- **Recommendation:**
  - **KEEP** - This is actively used in the core API
  - It's a small, focused module with a clear purpose
  - Part of the current architecture (Phase 3/4)

### 5. `hologram/smi.py` (MINIMAL USE)
- **Status:** ⚠️ **Only defines `MemoryPacket` dataclass**
- **Usage:**
  - Used by `api.py` for `retrieve()` method
  - Tested in `test_phase4_retrieval.py`
  - Clean, focused module (62 lines)
- **Recommendation:**
  - **KEEP** `MemoryPacket` - it's actively used
  - Consider merging into `retrieval.py` since they're related (both Phase 4 components)

### 6. `hologram/retrieval.py` (MINIMAL USE)
- **Status:** ⚠️ **Only used by `api.py` and tested**
- **Purpose:** Extract local gravitational field for probe-based retrieval
- **Details:**
  - Exports `extract_local_field()` function
  - Used in `Hologram.retrieve()` method
  - Part of Phase 4: Dynamic Graph Retrieval
- **Recommendation:**
  - **KEEP** - This is part of the active retrieval system
  - Well-integrated with the current architecture

---

## 🟢 LOW PRIORITY - Modules with Indirect Usage

### 7. `hologram/embeddings.py` (INDIRECT USE)
- **Status:** ✅ **Used indirectly through `api.py`**
- **Details:**
  - Imported by `api.py` which is used by 23 files
  - Core embedding functionality
- **Recommendation:** **KEEP**

### 8. `hologram/glyphs.py` (INDIRECT USE)
- **Status:** ✅ **Used indirectly through `api.py`**
- **Details:**
  - `GlyphRegistry` is instantiated in `Hologram.init()`
  - Core memory organization component
- **Recommendation:** **KEEP**

---

## 📊 Import Statistics

| Module | Direct Imports | Status |
|--------|---------------|--------|
| `api.py` | 23 | ✅ Core |
| `text_utils.py` | 7 | ✅ Core |
| `gravity.py` | 6 | ✅ Core |
| `chatbot.py` | 4 | ✅ Active |
| `mg_scorer.py` | 3 | ✅ Active |
| `store.py` | 3 | ✅ Core |
| `manifold.py` | 1 | ⚠️ Minimal |
| `retrieval.py` | 1 | ⚠️ Minimal |
| `smi.py` | 1 | ⚠️ Minimal |
| `demo_clip.py` | 1 | ⚠️ Demo only |
| `config.py` | 0* | ⚠️ Indirect |
| `embeddings.py` | 0* | ✅ Indirect |
| `glyphs.py` | 0* | ✅ Indirect |
| `run_sim.py` | 0 | ❌ Unused |

*Imported indirectly through other modules

---

## 🎯 Recommended Actions

### Immediate (High Priority)
1. **DELETE** `hologram/run_sim.py` - No longer used, outdated paths
2. **CONSOLIDATE** `hologram/demo_clip.py` → `demos/demo_clip.py`

### Short-term (Medium Priority)
3. **CONSIDER** merging `smi.py` into `retrieval.py` (both are Phase 4 retrieval components)
4. **CLEAN UP** `hologram/api.py` - Remove commented-out imports (lines 5, 326-328)

### Long-term (Low Priority)
5. **EXPAND** `hologram/config.py` with more configuration options
6. **DOCUMENT** the purpose of minimal-use modules (`manifold`, `retrieval`, `smi`) in architecture docs

---

## 📁 File Organization Suggestions

### Current Structure Issues
- Core library modules mixed with demo code (`demo_clip.py`)
- Standalone scripts in core package (`run_sim.py`)

### Suggested Structure
```
hologram/
├── api.py              # Main API (keep)
├── gravity.py          # Physics engine (keep)
├── embeddings.py       # Encoders (keep)
├── glyphs.py           # Memory organization (keep)
├── store.py            # Storage backend (keep)
├── text_utils.py       # Text processing (keep)
├── chatbot.py          # Chat interface (keep)
├── mg_scorer.py        # MG scoring (keep)
├── config.py           # Configuration (keep, expand)
├── manifold.py         # Vector alignment (keep)
├── retrieval.py        # Probe retrieval (keep, maybe merge with smi.py)
├── smi.py              # Memory packets (keep, clean up)
└── storage/            # Storage implementations (keep)

demos/                  # Move all demos here
├── demo.py
├── demo_clip.py        # ← Merge hologram/demo_clip.py here
├── demo_decay.py
└── ...

scripts/                # Standalone utilities
└── (no changes needed)
```

---

## 🔍 Notes on Architecture

Based on conversation history, the current architecture includes:
- **Phase 3:** Latent Manifold integration (active)
- **Phase 4:** Dynamic Graph Retrieval with probes (active)
- **Dynamic Gravity System:** Auto-fusion and auto-mitosis (recently implemented)
- **MG Scorer:** Semantic coherence metrics (recently integrated)

The minimal-use modules (`manifold`, `retrieval`, `smi`) are **intentional architectural components** from recent phases, not legacy code. They should be kept but better documented.

---

## ✅ Conclusion

**Files to Remove:**
1. `hologram/run_sim.py` (unused prototype)
2. `hologram/demo_clip.py` (consolidate into demos/)

**Files to Clean:**
1. `hologram/api.py` (remove commented-out imports on lines 5, 326-328)

**Files to Keep:**
All other modules are either core components or actively used in the current architecture.

**Total Cleanup Impact:** 2 files deleted, 1 file cleaned = cleaner, more maintainable codebase

