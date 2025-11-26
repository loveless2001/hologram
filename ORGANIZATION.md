# Directory Organization - Nov 23, 2023

## Summary of Changes

The project has been reorganized for better clarity and maintainability.

## New Structure

### Root Directory (Clean!)
Now contains only:
- **Core scripts**: `chat_cli.py`, `web_ui.py`, `run_ui.sh`
- **Documentation**: `README.md`, `QUICKREF.md`, `RELATIONS.md`
- **Directories**: `hologram/`, `api_server/`, `demos/`, `tests/`, `scripts/`, `data/`, `docs/`

### `/demos` Directory
**Purpose**: Demonstration scripts showcasing features

**Contents** (7 scripts):
- `demo.py` - Basic text-only usage
- `demo_clip.py` - Text → image search
- `demo_img2img.py` - Image → image similarity
- `demo_negation.py` - Negation-aware gravity
- `demo_decay.py` - Concept reinforcement & decay
- `demo_knowledge_base.py` - KB construction
- `demo_kg_comparison.py` - vs traditional KGs

**README**: `demos/README.md` with usage instructions

### `/tests` Directory
**Purpose**: All test scripts and validation

**Contents** (13 items):
- **Unit tests**: `test_chatbot.py`, `test_chaos.py`, `test_chaos_viz.py`
- **Integration tests**: `test_api.py`, `test_faiss.py`, `test_gliner.py`
- **Feature tests**: `test_reconstruction.py`, `test_relations.py`, `test_search_relations.py`
- **Utilities**: `benchmark.py`, `check_cuda.py`
- **Subdirectory**: `integration/` (for future organized integration tests)

**README**: `tests/README.md` with test categories and running instructions

### `/data` Directory
**Purpose**: Generated files and test data

**Moved here**:
- `philosophy_memory.json` (generated memory snapshot)
- `relativity_kb.json` (generated KB snapshot)
- `benchmark_results.json` (benchmark output)
- `output.png` (visualization output)

**Existing**:
- `kbs/` - Knowledge base text files
- Images for CLIP demos

### Root Scripts (Unchanged)
- `chat_cli.py` - CLI chat interface
- `web_ui.py` - Streamlit UI
- `run_ui.sh` - Quick launcher

## Benefits

### 1. Cleaner Root
- 13 files/dirs → 7 core files + 7 directories
- Easier to navigate
- Clear separation of concerns

### 2. Better Organization
- **Demos** clearly separated from **Tests**
- Easy to find examples
- Test suite well-organized

### 3. Improved Documentation
- Each directory has its own README
- Clear usage instructions
- Better onboarding for new developers

### 4. Scalability
- Easy to add new demos
- Test organization supports growth
- Data/output files contained

## Migration Guide

### Old → New Paths

**Demos**:
```bash
# Before
python demo.py
python demo_negation.py

# After
python demos/demo.py
python demos/demo_negation.py
```

**Tests**:
```bash
# Before
python test_reconstruction.py
python test_search_relations.py

# After
python tests/test_reconstruction.py
python tests/test_search_relations.py
```

**Data**:
```bash
# Before
./relativity_kb.json
./benchmark_results.json

# After  
./data/relativity_kb.json
./data/benchmark_results.json
```

## No Breaking Changes

- All Python imports unchanged (package paths intact)
- API server still runs from root
- Tests still discoverable by pytest
- Documentation updated to reflect new paths

## Files Updated

1. `README.md` - Repository layout section + demo paths
2. `demos/README.md` - New comprehensive demo guide
3. `tests/README.md` - New test organization guide
4. This file - Organization summary

---

**Status**: ✅ Complete
**Date**: 2023-11-26
**Impact**: Low (documentation only, no code changes needed)

## Update (Nov 26, 2023)
- **Refactoring**: Moved concept logic from `ChatMemory` to `Hologram` and `GravityField`.
- **New Features**: Added **Concept Mitosis** (contextual disambiguation) and **Graph-Based Reconstruction**.
- **Impact**: `ChatMemory` is thinner; core logic is now universal for all text inputs.
