# HoloChat Overview

HoloChat is a chatbot app built on top of Hologram retrieval. It lives at `apps/holochat/` inside this repo so Hologram stays a pure retrieval engine and the chat surface can evolve independently.

## What it does

Takes a user message + optional session/document context, classifies it into one of 8 intents, picks a retrieval scope, runs the appropriate pipeline, and returns a grounded reply with citations.

## Architecture

```
 user message + context
          |
          v
   [ RouterContext ]
          |
          v
   [ Router ]   rules → prototype classifier → (LLM router stub, off by default)
          |
          v
   [ RouteDecision ]   intent + scope + response_mode + needs_retrieval/rerank
          |
          v
   [ Orchestrator ]
          |
          |--> [ RetrievalAdapter ] ---> Hologram.search_scoped(...)
          |         (if needs_retrieval)
          |
          |--> [ SessionStore ]         (if intent == session_memory)
          |
          v
   [ Provider ]     OpenAI if OPENAI_API_KEY set, else deterministic reply
          |
          v
   [ ChatResponse ]   reply + citations + route metadata
```

## Module map

| Module | Responsibility |
|---|---|
| `apps/holochat/schemas.py` | `RouterContext`, `RouteDecision`, `ChatRequest`, `ChatResponse`, `Citation` |
| `apps/holochat/router/rules.py` | Deterministic regex + context rules (layer 1) |
| `apps/holochat/router/prototypes.py` | Seed utterances + cached MiniLM centroids |
| `apps/holochat/router/classifier.py` | Embedding prototype classifier (layer 2) |
| `apps/holochat/router/decision.py` | Merge rule + classifier → final `RouteDecision` |
| `apps/holochat/router/llm_router.py` | Stub for LLM fallback (disabled in v1) |
| `apps/holochat/retrieval_adapter.py` | Maps scope → `search_scoped(...)` call |
| `apps/holochat/session_store.py` | In-memory session messages + summary + keyword-overlap search |
| `apps/holochat/providers.py` | `ChatProvider` protocol + OpenAI implementation |
| `apps/holochat/orchestrator.py` | Glue: hydrate context → route → retrieve → reply |
| `apps/holochat/server.py` | FastAPI app with `/chat/route` and `/chat/respond` |

## Integration seam into Hologram

The only coupling into Hologram beyond reading `Trace.meta` is `Hologram.search_scoped(query, *, glyph_ids, doc_ids, trace_filter, mode, ...)` in `hologram/api.py`. Everything else in `apps/holochat/` is self-contained.

## V1 scope

- 8 intents: `chat`, `knowledge_qa`, `doc_qa`, `summarization`, `comparison`, `session_memory`, `task_action`, `fallback`
- Router layers: rules → prototype classifier (LLM fallback stubbed, off)
- In-memory session store (SQLite deferred)
- Single FastAPI app at port 8011
- No auth, no frontend (use curl / HTTPie / the integration tests)

## Running

```bash
# From repo root
uvicorn apps.holochat.server:app --host 127.0.0.1 --port 8011
```

Or the module shortcut: `python -m apps.holochat.server` (reads `OPENAI_API_KEY` from `.env` if present; falls back to deterministic replies otherwise).

## Testing

- Unit tests for the scoped retrieval helper: `tests/test_search_scoped_retrieval_helper.py`
- End-to-end integration tests: `tests/test_holochat_integration_end_to_end.py`
- Codex's router unit tests live under `apps/holochat/router/tests/` (own-module-own-tests convention)

Run just the holochat suite:

```bash
./.venv/bin/pytest -q tests/test_search_scoped_retrieval_helper.py tests/test_holochat_integration_end_to_end.py
```

## See also

- [API reference](api-reference.md)
- [Router intents](router-intents.md)
- [Extending](extending.md)
