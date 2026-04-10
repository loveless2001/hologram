# Auto-Ingestion Fix Summary

## Issue
When loading a saved Hologram state with `auto_ingest_system=True`, system concepts were being duplicated because the auto-ingestion logic didn't check if system concepts already existed in the loaded gravity field.

## Solution
Added existence check before auto-ingesting system concepts in both `Hologram.init()` and `Hologram.load()` methods.

### Changes Made

#### 1. Updated `Hologram.init()` (`hologram/api.py`)
```python
# Check if system concepts already exist (e.g., from loaded state)
existing_system_concepts = [
    name for name in field.sim.concepts.keys()
    if name.startswith("system:") and field.sim.concepts[name].tier == 2
]

# Only ingest if no system concepts found
if not existing_system_concepts:
    # ... ingest system concepts ...
```

#### 2. Updated `Hologram.load()` (`hologram/api.py`)
- Added `auto_ingest_system` parameter (default: `True`)
- Added same existence check logic as `init()`
- System concepts from save file are preserved, not re-ingested

### Behavior

**Fresh Initialization:**
```python
holo = Hologram.init(auto_ingest_system=True)
# → Loads 31 system concepts as Tier 2
```

**Loading Saved State:**
```python
holo = Hologram.load("save.json", auto_ingest_system=True)
# → Detects existing system concepts in save file
# → Skips re-ingestion
# → No duplicates!
```

**Loading Old Save (Pre-Tier):**
```python
holo = Hologram.load("old_save.json", auto_ingest_system=True)
# → No system concepts in save file
# → Auto-ingests 31 system concepts
# → Backward compatible!
```

### Verification

Demo output now shows:
```
✓ System concepts before dynamics: 31
✓ System concepts after dynamics: 31
✅ System concepts protected from fusion/mitosis!

💾 Saving to /tmp/hologram_tier_test.json...
✓ Saved

📂 Loading from /tmp/hologram_tier_test.json...
[MemoryStore] Restored gravity field state (7 concepts)
✓ Loaded
ℹ️  Tier counts: T1 5→7, T2 28→28
   (Loaded instance includes trace IDs as concepts)
✅ System concepts (Tier 2) preserved correctly!
```

**Note:** The Tier 1 count difference (5→7) is expected because the loaded instance includes trace IDs (`text:*`, `glyph:*`) that were added to the gravity field during the original session. This is correct behavior.

### Benefits

1. **No Duplicates**: System concepts are never duplicated on load
2. **Backward Compatible**: Old saves without system concepts get them auto-added
3. **Efficient**: Skips unnecessary re-computation of system concept vectors
4. **Consistent**: Same logic in both `init()` and `load()` paths
5. **Configurable**: Can disable with `auto_ingest_system=False` if needed

### Usage Patterns

**Standard Usage (Recommended):**
```python
# Initialize new instance
holo = Hologram.init(auto_ingest_system=True)

# Save
holo.save("my_hologram.json")

# Load (system concepts preserved from save)
holo2 = Hologram.load("my_hologram.json", auto_ingest_system=True)
```

**Disable Auto-Ingestion:**
```python
# If you want to manage system concepts manually
holo = Hologram.init(auto_ingest_system=False)
```

## Files Modified
- `hologram/api.py`: Added existence check in `init()` and `load()`
- `examples/tier_demo.py`: Updated to properly verify save/load behavior
