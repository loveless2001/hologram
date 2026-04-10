# Phase 3: API Integration

## Status: TODO
## Priority: MEDIUM

## Overview
Wire GlyphRouter into existing API and server. New `/query/routed` endpoint runs parallel to `/query` for A/B comparison. All retrieval goes through GlyphOperator transforms.

## Related Code Files

### Modify
- `hologram/api.py` — `Hologram.__init__()`: instantiate GlyphRouter; add `search_routed()` method
- `hologram/api.py` — `Hologram.add_text()`: call `router.invalidate()` after trace attach
- `hologram/server.py` — add `/query/routed` endpoint

### Read
- `hologram/glyph_router.py` — GlyphRouter from Phase 2
- `hologram/glyph_operator.py` — GlyphOperator from Phase 1

## Implementation Steps

1. In `Hologram.__init__()` or `Hologram.init()`: create `self.router = GlyphRouter(store, glyphs, gravity_field)`
2. Add `Hologram.search_routed(query, top_k, top_glyphs)`:
   - Encode query via manifold
   - Delegate to `self.router.search_routed()`
   - Return trace + score pairs (same format as `search_text()`)
3. In `Hologram.add_text()`, after attaching trace: call `self.router.invalidate()`
4. Add `/query/routed` endpoint in server.py:
   - Same request schema as `/query`
   - Calls `holo.search_routed()`
   - Returns same response format

## Key Decisions
- New endpoint alongside existing — no breaking changes
- `search_with_drift()` unchanged — probe physics still uses global field
- Router invalidation on every `add_text()`

## Success Criteria
- [ ] `Hologram.search_routed()` works from Python API
- [ ] `/query/routed` endpoint returns results
- [ ] Existing `/query` endpoint unchanged
- [ ] No breaking changes to current consumers
