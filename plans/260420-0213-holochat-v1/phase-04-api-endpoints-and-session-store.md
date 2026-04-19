# Phase 04 — FastAPI Endpoints + Session Store

**Owner:** codex
**Status:** pending
**Priority:** P1
**Depends on:** Phase 03

## Overview

Expose `respond(...)` over HTTP. Persist per-session state (active_doc, selected_docs, recent_messages, summary) in SQLite.

## Endpoints

- `POST /chat/session` → create session, return `session_id`
- `POST /chat/session/{session_id}/context` → set `active_document_id` / `selected_document_ids`
- `POST /chat/route` → return `RouteDecision` only (no pipeline execution; for debugging/UI)
- `POST /chat/respond` → full pipeline, return `ChatResponse`
- `GET /chat/session/{session_id}` → session state + last N messages

Ingest endpoints are NOT duplicated — existing `/ingest/file` on hologram server handles that.

## Session Store

**V1: in-memory** (dict-backed), co-located in app dir: `apps/holochat/session_store.py`. SQLite schema below is the phase-2 hardening target.

SQLite schema (deferred to post-v1):

```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL,
    active_document_id TEXT,
    selected_document_ids TEXT,  -- JSON array
    session_summary TEXT
);
CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    role TEXT,          -- "user" | "assistant"
    content TEXT,
    intent TEXT,
    created_at REAL,
    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
);
```

Recent messages (last 3–5 turns) loaded into `RouterContext.recent_messages` on each `/chat/respond` call.

## Mounting Strategy

Option A: separate FastAPI app in `apps/holochat/server.py`, run on its own port
Option B: include chat router in existing `hologram/server.py` via `app.include_router(chat_router, prefix="/chat")`

**Recommend Option A for v1** — keeps boundaries clean; user can reverse-proxy if unified deployment is needed later.

## Files to Create

- `apps/holochat/server.py`
- `apps/holochat/session_store.py`
- `tests/test_holochat_server.py` (unit-level: endpoint smoke via TestClient)

## Todo

- [ ] SQLite schema + migration (run on first startup)
- [ ] `SessionStore` class with create/get/update/append_message
- [ ] FastAPI app with 5 endpoints
- [ ] Wire `/chat/respond` → orchestrator → response
- [ ] TestClient smoke tests for each endpoint

## Success Criteria

- All 5 endpoints return 200 on happy paths
- Session state persists across requests
- SQLite file auto-created at `data/holochat.db` on first startup
- Server runs via `uvicorn apps.holochat.server:app`
