# Phase 05 — Integration Tests + Docs

**Owner:** claude
**Status:** pending
**Priority:** P0 (gating release)
**Depends on:** Phases 01–04

## Overview

End-to-end tests that exercise the full router→adapter→orchestrator→API flow with real (not mocked) hologram, and docs that explain how to run / extend the chatbot.

## Integration Test File

`tests/test_holochat_integration.py`

Fixtures:
- Fresh `Hologram` instance with 2 small in-memory PDFs ingested under one glyph
- `EchoProvider` (deterministic, no external API calls)
- FastAPI `TestClient` for `apps.holochat.server:app`

Test cases:

1. **chat intent** — greeting goes through, no retrieval, no citations
2. **knowledge_qa** — question over corpus, grounded answer with citations from both docs
3. **doc_qa with active_document_id** — citations only from the selected doc; second doc never appears
4. **doc_qa with selected_document_ids (multi)** — citations only from selected subset
5. **fallback low-confidence** — vague query still returns a cautious grounded response
6. **session persistence** — two-turn conversation; second turn sees recent_messages populated
7. **`/chat/route` only** — returns RouteDecision without executing pipeline
8. **doc_ids scope correctness** — ingest doc A + doc B into same glyph; verify `search_scoped(doc_ids=[A])` never returns B's traces

## Docs

Create in `docs/holochat/`:

- `docs/holochat/overview.md` — what it is, architecture diagram (text), v1 scope
- `docs/holochat/api-reference.md` — all 5 endpoints with request/response schemas
- `docs/holochat/router-intents.md` — 5 intents, when each fires, scope mapping table
- `docs/holochat/extending.md` — how to add a new intent, new pipeline, new provider

Update root `README.md` with a short "See `docs/holochat/` for chatbot app" pointer.

## Todo

- [ ] Build fixtures for 2-doc corpus + EchoProvider
- [ ] Write all 8 integration test cases
- [ ] Verify doc_ids isolation under shared-glyph ingestion
- [ ] Write 4 docs files
- [ ] Update root README
- [ ] Run full test suite: existing + new router + new orchestrator + new integration → all pass

## Success Criteria

- All 8 integration tests pass
- Zero regressions in existing hologram test suite
- Docs cover: how to install, run, add intent, add pipeline
- `apps/holochat/README.md` at least lists entry points and invocation

## Risk / Notes

- Keep test corpus tiny to keep CI fast (< 1s ingest time)
- Use `EchoProvider` that echoes a canned response based on retrieved trace count — avoids flakiness from LLM calls
