# Phase 03 — Orchestrator: Intent → Pipeline Dispatch

**Owner:** codex
**Status:** pending
**Priority:** P0 (blocks phase 04)
**Depends on:** Phase 01 (router), Phase 02 (retrieval adapter)

## Overview

Single entry point `respond(ctx) -> ChatResponse` that: runs router → runs retrieval via adapter → optional rerank → calls provider with built prompt → returns grounded answer + citations.

## Module Layout

```
apps/holochat/
├── orchestrator.py       # main respond() + pipeline mapping
├── pipelines.py          # per-intent pipelines: chat, knowledge_qa, doc_qa, task_action, fallback
├── providers.py          # ChatProvider protocol (reuses hologram.chatbot if possible)
└── tests/
    └── test_orchestrator.py
```

## Pipeline Contract

Each pipeline takes `(ctx, decision, hologram, provider)` and returns `ChatResponse`:

```python
class ChatResponse(BaseModel):
    reply: str
    intent: str
    confidence: float
    citations: list[Citation] = []  # trace_id, source_doc, score, snippet
    used_retrieval: bool = False
    retrieval_scope: str = "none"
    action_result: Optional[dict] = None  # for task_action
```

## Per-Intent Pipelines

- **chat:** provider call with recent history, no retrieval
- **knowledge_qa:** `retrieve_for_decision` → rerank if `needs_rerank` → build grounded prompt with citations → provider call
- **doc_qa:** same as knowledge_qa but scope already set to active/selected docs
- **task_action:** dispatch to `actions/` registry (v1: stubbed actions, log-only); no provider call unless confirmation text needed
- **fallback:** broad `knowledge_qa`-like pipeline with lower retrieval threshold + cautious prompt

## Reranking

Stub interface in v1: `rerank(traces, query) -> traces` — default implementation is identity (no-op) with a TODO for cross-encoder reranker in v2. Keeps the seam but no new model dependency.

## Files to Create

- `apps/holochat/orchestrator.py`
- `apps/holochat/pipelines.py`
- `apps/holochat/providers.py` (thin wrapper / re-export from `hologram.chatbot`)
- `apps/holochat/schemas.py` additions: `ChatResponse`, `Citation`
- `tests/test_orchestrator.py`

## Todo

- [ ] Define `ChatResponse` / `Citation` schemas
- [ ] Implement `pipelines.py` with 5 pipelines
- [ ] Implement `orchestrator.respond(ctx)` that wires router + adapter + pipeline
- [ ] Stub `rerank()` with no-op + TODO
- [ ] Unit tests with mocked provider + mocked hologram

## Success Criteria

- `respond(ctx)` routes and returns `ChatResponse` for all 5 intents
- Citations populated from retrieved traces for retrieval-bearing intents
- Unit tests pass with mocked dependencies
