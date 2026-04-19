# Phase 06 — Optional Static HTML/JS Smoke UI

**Owner:** claude
**Status:** optional / stretch
**Priority:** P2 (ship only if time permits and manual testing demands it)
**Depends on:** Phase 04

## Overview

Single-page static HTML + vanilla JS that talks to `/chat/respond` and renders the conversation. No build step, no framework, no npm. Ship only if codex's JSON endpoints aren't enough for manual testing.

## Scope

- Single file: `apps/holochat/web/index.html` (HTML + inline JS + minimal CSS)
- Features:
  - Text input + send button
  - Streaming display of assistant reply (optional; plain POST fine for v1)
  - Toggle for `active_document_id` (dropdown populated from a `GET /chat/docs` endpoint if added)
  - Session state shown at top (session_id, active_doc)
  - Citations rendered under each assistant reply
- Served via FastAPI `StaticFiles` mount in `apps/holochat/server.py`

## Files

- `apps/holochat/web/index.html`
- `apps/holochat/web/style.css` (optional; inline if small)

## Todo

- [ ] Decide if it's actually needed once phase 04 lands (curl/HTTPie may be enough)
- [ ] If yes: single-file HTML + JS with vanilla fetch
- [ ] Mount at `/web/` in server
- [ ] Smoke-test in browser: send message, verify response + citations render

## Success Criteria

- Opens at `http://localhost:8000/web/` and sends a round-trip message without console errors

## When to Skip

Skip entirely if:
- Curl/HTTPie suffices for integration testing
- User wants to build their own frontend and this would be throwaway
- Phase 05 integration tests provide enough confidence
