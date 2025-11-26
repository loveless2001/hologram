# Test Scripts

This directory contains test scripts for validation, integration testing, and feature verification.

## Test Organization

### Unit Tests (`tests/test_*.py`)
- **`test_chatbot.py`** - Chat memory and session tests (pytest)
- **`test_chaos.py`** - Chaos testing for stability
- **`test_chaos_viz.py`** - Visualization chaos testing

### Integration Tests
- **`test_api.py`** - API endpoint validation
- **`test_faiss.py`** - FAISS vector index tests
- **`test_gliner.py`** - GLiNER concept extraction verification
- **`test_reconstruction.py`** - Knowledge reconstruction from seed keywords
- **`test_relations.py`** - Concept relation extraction tests
- **`test_search_relations.py`** - Enhanced search endpoint with relations

### Utility Scripts
- **`benchmark.py`** - Performance benchmarking
- **`check_cuda.py`** - CUDA availability check

## Running Tests

### Run All Unit Tests
```bash
source .venv/bin/activate
pytest
```

### Run Specific Tests
```bash
# Unit tests
pytest tests/test_chatbot.py
pytest tests/test_chaos.py

# Integration tests
python tests/test_api.py
python tests/test_gliner.py
python tests/test_reconstruction.py
python tests/test_search_relations.py
```

### Run Benchmarks
```bash
python tests/benchmark.py
```

## Test Categories

### 1. Core Functionality
- Memory storage and retrieval
- Vector encoding
- Gravity field mechanics

### 2. API Endpoints
- `/search` with relations
- `/chat` conversational interface
- `/viz-data` visualization endpoint

### 3. Concept Extraction
- GLiNER model loading
- Sentence → atomic concepts
- Relation/verb detection
- Order preservation

### 4. Knowledge Reconstruction
- Seed keyword → related concepts
- Semantic clustering
- Context retrieval

### 5. Stability
- Chaos testing (random inputs)
- Edge cases
- Error handling

## Requirements

**Basic tests**:
```bash
pip install pytest numpy faiss-cpu
```

**Full test suite**:
```bash
pip install pytest numpy faiss-cpu gliner transformers requests
```

**With CLIP**:
```bash
pip install torch torchvision open_clip_torch
```

## Test Data

Some tests use:
- `data/kbs/relativity.txt` - Sample knowledge base
- `data/cat.png`, `data/dog.png` - Sample images (for CLIP tests)

## Expected Results

All tests should pass with current implementation. If tests fail:
1. Check dependencies are installed
2. Ensure API server is running (for API tests)
3. Verify test data files exist

## Adding New Tests

1. Create test file: `tests/test_<feature>.py`
2. Follow pytest conventions for unit tests
3. Use standalone scripts for integration tests
4. Update this README with test description

## Continuous Integration

Tests are designed to run in CI/CD pipelines:
```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    pip install -r requirements.txt
    pytest
```
