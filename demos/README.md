# Demo Scripts

This directory contains demonstration scripts showcasing various hologram features.

## Available Demos

### Core Features
- **`demo.py`** - Text-only walkthrough (basic usage)
- **`demo_clip.py`** - Text → image search (requires CLIP)
- **`demo_img2img.py`** - Image → image similarity (requires CLIP)

### Advanced Features
- **`demo_negation.py`** - Negation-aware gravity field
- **`demo_decay.py`** - Reinforcement-based concept decay
- **`demo_knowledge_base.py`** - Knowledge base construction and querying
- **`demo_kg_comparison.py`** - Comparison with traditional knowledge graphs

## Running Demos

```bash
# From project root
source .venv/bin/activate

# Basic demo (no dependencies)
python demos/demo.py

# Negation demo
python demos/demo_negation.py

# Knowledge base demo
python demos/demo_knowledge_base.py

# CLIP demos (requires torch + open_clip)
python demos/demo_clip.py
python demos/demo_img2img.py
```

## Requirements

**All demos**:
- numpy
- faiss-cpu (or faiss-gpu)

**CLIP demos only**:
- torch
- torchvision
- open_clip_torch
- Pillow

## Demo Order (Recommended)

1. `demo.py` - Understand basic concepts
2. `demo_negation.py` - See how negation affects gravity
3. `demo_decay.py` - Learn about concept reinforcement
4. `demo_knowledge_base.py` - Build domain-specific KBs
5. `demo_clip.py` - Explore text ↔ image search
6. `demo_img2img.py` - Image similarity search

## Output

Demos print to console and some may generate visualizations or save memory snapshots.
