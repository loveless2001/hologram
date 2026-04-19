# HoloChat v1 — Intent-Routed Chatbot on Hologram

## Overview

A minimal chatbot app that uses Hologram as a retrieval engine. Routes each user turn through a lightweight intent classifier and dispatches to a scoped retrieval + response pipeline. Built in-repo at `apps/holochat/` to keep Hologram reusable and avoid turning `hologram/chatbot.py` into a second monolith.

## V1 Scope

**Intents (8):** `chat`, `knowledge_qa`, `doc_qa`, `summarization`, `comparison`, `session_memory`, `task_action`, `fallback`
**Scopes (5):** `none`, `global_corpus`, `active_document`, `selected_documents`, `session_memory`
**Response modes (6):** `chat`, `grounded_answer`, `summary`, `comparison`, `action`, `fallback`
**Router layers (2 + 1 stub):** deterministic rules → embedding prototype classifier (MiniLM) → LLM router stub (interface only, disabled by default; v2 fills in)
**Session store:** in-memory for v1 (fast iteration); SQLite hardening deferred to post-v1

Note: earlier draft reduced to 5 intents on YAGNI grounds. Revised to 8 to match owner's implementation — full intent set lets us validate router behavior against the full user spec rather than a synthetic subset.

## Defaults (ratified in #hologram on 2026-04-20)

- Location: `apps/holochat/` in this repo
- Provider: OpenAI-first via existing `OpenAIChatProvider` abstraction; clean interface so Anthropic/local can slot in later
- Transport: FastAPI JSON endpoints; minimal static HTML/JS smoke UI only if needed for manual testing (no React)
- Auth: single-user local, none in v1

## Architecture Boundaries

- **Hologram** stays the engine: ingest + retrieval only
- **HoloChat** owns: routing, orchestration, session state, API
- One narrow seam from chatbot into engine: `search_scoped(...)` added to `hologram/api.py`
- No chat orchestration creeps into `hologram/api.py` beyond that helper

## Phases

| # | Phase | Owner | Status | Notes |
|---|---|---|---|---|
| 01 | Router schemas + rules + prototype classifier | codex | in-flight | 8 intents, LLM router stub off by default |
| 02a | `search_scoped()` helper in `hologram/api.py` | codex | landed | exact cosine over filtered candidates |
| 02b | Retrieval adapter in `apps/holochat/` | codex | landed | absorbed by codex during parallel work |
| 03 | Orchestrator: intent→pipeline dispatch | codex | in-flight | |
| 04 | In-memory session store + FastAPI endpoints | codex | in-flight | SQLite hardening deferred |
| 05 | Integration tests + docs | claude | pending | covers phases 01–04 as they land |
| 06 | Optional static HTML/JS smoke UI | claude | optional | ship only if manual testing demands |

**Revised ownership:** codex absorbed phases 02b (adapter) and 04 (session store) during parallel execution. Claude's remaining scope is phase 05 (integration tests + docs) and phase 06 (optional UI). Claude has also landed unit tests for `search_scoped` at `tests/test_search_scoped_retrieval_helper.py` (17 tests passing).

## Key Contracts

**RouteDecision** (produced by router, consumed by orchestrator) — see `phase-01`
**search_scoped(...)** (chatbot→engine seam) — see `phase-02`

## Test Strategy

- **Unit tests:** each owner writes unit tests for their own modules (parser-split precedent)
- **Integration tests:** claude owns end-to-end chat-flow and scoped-retrieval-semantics tests
- All tests run under existing `tests/` infra; new integration file: `tests/test_holochat_integration.py`

## Success Criteria

- `POST /chat/respond` with `user_message` + `active_document_id=None` routes to `knowledge_qa`, retrieves via `search_dynamic`, returns grounded answer
- Same endpoint with `active_document_id=<doc>` routes to `doc_qa`, retrieves only from that doc via `doc_ids=[...]`
- `POST /chat/route` returns a `RouteDecision` without executing a pipeline (useful for debugging/UI)
- Router unit tests pass for all 5 intents, both rule-path and classifier-path
- Integration tests cover: chat-only (no retrieval), knowledge_qa grounded, doc_qa scoped, fallback behavior

## Phase Files

- [Phase 01 — Router](phase-01-router-schemas-rules-classifier.md) (codex, in-flight)
- [Phase 02 — Scoped Retrieval](phase-02-scoped-retrieval-helper-and-adapter.md) (codex, landed — both helper and adapter)
- [Phase 03 — Orchestrator](phase-03-orchestrator-intent-to-pipeline.md) (codex, in-flight)
- [Phase 04 — API + Sessions](phase-04-api-endpoints-and-session-store.md) (codex, in-flight; in-memory store for v1)
- [Phase 05 — Integration Tests + Docs](phase-05-integration-tests-and-docs.md) (claude, pending)
- [Phase 06 — Smoke UI](phase-06-optional-static-smoke-ui.md) (claude, optional)
