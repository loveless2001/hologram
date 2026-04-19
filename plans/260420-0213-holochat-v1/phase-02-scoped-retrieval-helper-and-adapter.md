# Phase 02 — Scoped Retrieval Helper + Adapter

**Owner:** claude
**Status:** pending (can start immediately; parallel with phase 01)
**Priority:** P0 (blocks phase 03)

## Overview

Add one narrow helper to `hologram/api.py` that the chatbot orchestrator calls for all retrieval, regardless of intent scope. Then build a thin adapter in `apps/holochat/` that maps `RouteDecision` to `search_scoped(...)` calls.

## Key Contract — `search_scoped(...)`

Added to `Hologram` class in `hologram/api.py`:

```python
def search_scoped(
    self,
    query: str,
    *,
    top_k: int = 5,
    glyph_ids: Optional[List[str]] = None,    # shard narrowing (domain scoping)
    doc_ids: Optional[List[str]] = None,      # doc-level filter via trace.meta['source_doc']
    trace_filter: Optional[Callable[[Trace], bool]] = None,  # escape hatch for page/section/etc
    mode: str = "dynamic",                    # "global" | "global_pca" | "routed" | "dynamic"
    top_glyphs: int = 2,                      # only used when mode="routed" or dynamic→routed
) -> List[Tuple[Trace, float]]:
    ...
```

**Behavior:**
- `glyph_ids` narrows shards *before* retrieval when engine supports it (routed mode); otherwise applied as post-filter on glyph affinity
- `doc_ids` applied as post-filter on `trace.meta.get("source_doc")`
- `trace_filter` applied last, after `doc_ids`
- `mode="dynamic"` delegates to `choose_dynamic_strategy(...)` then filters results
- Result format matches existing `search_text()` / `search_routed()`: `List[Tuple[Trace, float]]`

**Non-behavior (out of scope for this helper):**
- No reranking — stays at orchestrator layer
- No query rewriting — caller owns that
- No format conversion — caller consumes raw traces

## Implementation Notes

Since hologram's existing search paths don't natively filter by `doc_ids`, `search_scoped` will:

1. Request `top_k * oversample` from underlying search (oversample=3 default)
2. Apply `doc_ids` + `trace_filter` filters
3. Truncate to `top_k`

This is cheap enough for v1; if oversample proves insufficient at scale, switch to shard-aware filtering in `glyphs.py` later.

For `glyph_ids`: in v1, if `mode in ("routed", "dynamic")` and `glyph_ids` non-None, restrict the router's candidate glyphs before shard search. If `mode in ("global", "global_pca")`, apply as post-filter on `trace.meta` or by cross-referencing the store.

## Adapter — `apps/holochat/retrieval_adapter.py`

Thin mapping layer. No retrieval logic of its own:

```python
def retrieve_for_decision(
    hologram: Hologram,
    decision: RouteDecision,
    ctx: RouterContext,
    top_k: int = 5,
) -> List[Tuple[Trace, float]]:
    if not decision.needs_retrieval:
        return []

    doc_ids = None
    if decision.retrieval_scope == "active_document" and ctx.active_document_id:
        doc_ids = [ctx.active_document_id]
    elif decision.retrieval_scope == "selected_documents":
        doc_ids = ctx.selected_document_ids or None

    return hologram.search_scoped(
        ctx.user_message,
        top_k=top_k,
        doc_ids=doc_ids,
        mode="dynamic",
    )
```

Reranking, if `decision.needs_rerank=True`, happens in the orchestrator (phase 03), not here.

## Files to Create / Modify

**Modify:**
- `hologram/api.py` — add `search_scoped(...)` method (~60 lines)

**Create:**
- `apps/holochat/retrieval_adapter.py`
- `tests/test_search_scoped.py` (unit tests for the helper)

## Todo

- [ ] Add `search_scoped(...)` to `Hologram` class with all four filter modes
- [ ] Unit tests covering: no filter (equivalent to existing paths), `doc_ids` filter, `glyph_ids` filter, `trace_filter` callable, combined filters, `mode` dispatch
- [ ] Implement `retrieval_adapter.py` with `retrieve_for_decision(...)`
- [ ] Smoke test: ingest 2 PDFs into same glyph, verify `doc_ids=[A]` returns only A's traces
- [ ] Run full existing test suite to confirm no regressions

## Dependencies

- Phase 01 schemas (`RouteDecision`, `RouterContext`) for the adapter — but schema shape is agreed; adapter can stub against the contract while codex lands the real schemas

## Success Criteria

- `search_scoped(query, doc_ids=[X])` returns only traces where `trace.meta["source_doc"] == X`
- All four `mode` values dispatch to the correct underlying search path
- `search_scoped()` with no filters produces identical results to `search_text()` (under `mode="global"`)
- Adapter correctly translates all 4 v1 scopes (`none`, `global_corpus`, `active_document`, `selected_documents`)
- Zero regressions in existing hologram test suite

## Risk / Notes

- Oversampling ratio (3x) is a guess; add a smoke test measuring recall loss on filtered queries and tune if needed
- If `doc_ids` filter yields fewer than `top_k` hits, return what we have rather than expanding the search — caller should handle short result sets
- Keep helper pure: no mutation of store, no caching of filtered results (would invalidate on ingest)
