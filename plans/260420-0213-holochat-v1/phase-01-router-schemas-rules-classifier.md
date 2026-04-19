# Phase 01 — Router Schemas + Rules + Prototype Classifier

**Owner:** codex
**Status:** pending (awaits user ratification of plan)
**Priority:** P0 (blocks phase 03)

## Overview

Build the intent router as three small modules that produce a single `RouteDecision` per user turn. No I/O, no retrieval, no LLM calls — pure logic.

## Key Contract — `RouteDecision`

```python
from pydantic import BaseModel
from typing import Literal, Optional

Intent = Literal[
    "chat", "knowledge_qa", "doc_qa",
    "summarization", "comparison", "session_memory",
    "task_action", "fallback",
]
RetrievalScope = Literal[
    "none", "global_corpus", "active_document",
    "selected_documents", "session_memory",
]
ResponseMode = Literal[
    "chat", "grounded_answer", "summary",
    "comparison", "action", "fallback",
]

class RouterContext(BaseModel):
    user_message: str
    active_document_id: Optional[str] = None
    selected_document_ids: list[str] = []
    recent_messages: list[str] = []
    session_summary: Optional[str] = None
    just_uploaded_files: bool = False

class RouteDecision(BaseModel):
    intent: Intent
    confidence: float
    needs_retrieval: bool
    retrieval_scope: RetrievalScope = "none"
    needs_rerank: bool = False
    needs_chat_history: bool = False
    response_mode: ResponseMode = "fallback"
    action_name: Optional[str] = None
```

Note: schema accommodates v2 intents/scopes (`summarization`, `comparison`, `session_memory`) by widening `Literal` later without breaking callers.

## Module Layout

```
apps/holochat/
├── __init__.py
├── router/
│   ├── __init__.py
│   ├── schemas.py        # RouteDecision, RouterContext
│   ├── rules.py          # deterministic rule engine
│   ├── prototypes.py     # example utterances per intent + cached centroids
│   ├── classifier.py     # embedding-based prototype classifier (uses hologram MiniLM)
│   └── decision.py       # merges rules + classifier → RouteDecision
└── tests/
    ├── test_router_rules.py
    ├── test_router_classifier.py
    └── test_router_decision.py
```

## Rules (deterministic, Layer 1)

Handle high-signal cases cheaply:
- Greetings / thanks / short casual → `chat` (no retrieval)
- "index this", "re-embed", "delete" → `task_action`
- "what does this doc say" + `active_document_id` present → `doc_qa`
- Pronoun-heavy follow-up without context → low confidence, defer to classifier

Rule output is a `partial RouteDecision` + confidence. Low confidence rules fall through to classifier.

## Classifier (embedding prototype, Layer 2)

- 10–30 seed utterances per intent in `prototypes.py`
- Embed via `hologram.embeddings.TextMiniLM` (same encoder hologram uses — no new model)
- Precompute centroids at module load
- At runtime: embed user message → cosine sim to each centroid → top-1 intent + confidence from top1/top2 margin

Confidence bands:
- `>= 0.80` high → use directly
- `0.60–0.79` medium → use but allow fallback retrieval broadening
- `< 0.60` low → `fallback` intent

## Merge Logic (`decision.py`)

```
def decide_route(ctx: RouterContext) -> RouteDecision:
    rule = apply_rules(ctx)
    if rule and rule.confidence >= 0.9:
        return rule
    clf = classify_with_embeddings(ctx)
    return merge(rule, clf, ctx)
```

Scope mapping after intent is set:
- `chat` → `none`, `response_mode="chat"`, `needs_chat_history=True`
- `knowledge_qa` → `global_corpus`, `grounded_answer`, `needs_rerank=True`
- `doc_qa` → `active_document` if `active_document_id` else `selected_documents` if `selected_document_ids` else downgrade to `knowledge_qa`
- `task_action` → `none`, `action`, populate `action_name`
- `fallback` → `global_corpus` (broad), `fallback`, `needs_rerank=False`

## Files to Create

- `apps/holochat/__init__.py`
- `apps/holochat/router/{__init__,schemas,rules,prototypes,classifier,decision}.py`
- `tests/test_router_rules.py`
- `tests/test_router_classifier.py`
- `tests/test_router_decision.py`

All Python modules use kebab-case-like-in-spirit but Python enforces snake_case for importability — filenames stay snake_case here, module names clear and descriptive.

## Dependencies

- `pydantic` (already in repo)
- `hologram.embeddings.TextMiniLM` (already available)

## Todo

- [ ] Create `apps/holochat/` package skeleton
- [ ] Implement `router/schemas.py`
- [ ] Implement `router/rules.py` with regex + context checks
- [ ] Curate 10–30 seed utterances per intent in `router/prototypes.py`
- [ ] Implement `router/classifier.py` with cached centroids
- [ ] Implement `router/decision.py` merge logic
- [ ] Unit tests: rules per intent, classifier confidence bands, decision merge edge cases
- [ ] Run `pytest -q tests/test_router_*.py` → all pass

## Success Criteria

- `decide_route(ctx)` returns valid `RouteDecision` for all 5 intents
- All unit tests pass; no regressions in existing hologram tests
- Classifier cold-start < 500ms (centroid precompute), per-query < 50ms

## Risk / Notes

- Prototype seed quality dominates classifier accuracy — treat seed curation as a deliverable, not an afterthought
- Keep `LLMRouter` stub interface in `decision.py` but disabled by default; v2 slots in without refactor
