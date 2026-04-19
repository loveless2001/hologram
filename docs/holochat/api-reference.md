# HoloChat API Reference

Two POST endpoints + a liveness root. All bodies are JSON, all responses validated by pydantic models defined in `apps/holochat/schemas.py`.

## `GET /`

Liveness check.

**Response:**
```json
{"status": "running", "service": "holochat", "version": "0.1.0"}
```

## `POST /chat/route`

Classifies the user message and returns a `RouteDecision` without executing any pipeline. Useful for UI previews and debugging intent classification.

**Request:**
```json
{
  "project": "my_project",
  "session_id": "optional-session-id",
  "context": {
    "user_message": "What does ETOPS certification require?",
    "active_document_id": null,
    "selected_document_ids": [],
    "recent_messages": [],
    "session_summary": null,
    "just_uploaded_files": false,
    "available_actions": [],
    "corpus_search_enabled": true
  }
}
```

**Response:**
```json
{
  "route": {
    "intent": "knowledge_qa",
    "confidence": 0.83,
    "needs_retrieval": true,
    "retrieval_scope": "global_corpus",
    "needs_rerank": true,
    "needs_chat_history": false,
    "response_mode": "grounded_answer",
    "action_name": null
  }
}
```

## `POST /chat/respond`

Full pipeline: route → retrieve → reply. Returns the `RouteDecision`, the assistant reply, and the citations used to ground it.

**Request:**
```json
{
  "project": "my_project",
  "session_id": "session-1",
  "context": { /* same RouterContext as /chat/route */ },
  "top_k": 5
}
```

**Response:**
```json
{
  "session_id": "session-1",
  "route": { /* RouteDecision */ },
  "reply": "Grounded answer: ...",
  "citations": [
    {
      "trace_id": "doc_a:chunk:0",
      "score": 0.74,
      "content": "ETOPS certification requires engine reliability ...",
      "source_doc": "doc_a.pdf",
      "page_number": 1,
      "meta": { /* full trace metadata */ }
    }
  ]
}
```

## Schema: `RouterContext`

| Field | Type | Default | Purpose |
|---|---|---|---|
| `user_message` | string | — | Required. The user turn. |
| `active_document_id` | string \| null | null | If set, `doc_qa`/`summarization` scope collapses to this doc. |
| `selected_document_ids` | string[] | [] | If set (and no active), scopes to this subset. |
| `recent_messages` | string[] | [] | Override recent turns; orchestrator will hydrate from session store if empty. |
| `session_summary` | string \| null | null | Optional rolling summary. |
| `just_uploaded_files` | bool | false | Hint used by rules for `task_action` vs `doc_qa`. |
| `available_actions` | string[] | [] | Names of actions the UI can execute. |
| `corpus_search_enabled` | bool | true | Disables retrieval globally when false. |

## Schema: `RouteDecision`

| Field | Type | Values |
|---|---|---|
| `intent` | literal | `chat`, `knowledge_qa`, `doc_qa`, `summarization`, `comparison`, `session_memory`, `task_action`, `fallback` |
| `confidence` | float | 0.0–1.0 |
| `needs_retrieval` | bool | — |
| `retrieval_scope` | literal | `none`, `global_corpus`, `active_document`, `selected_documents`, `session_memory` |
| `needs_rerank` | bool | reranking is stubbed in v1 (no-op) |
| `needs_chat_history` | bool | hints provider to include recent turns |
| `response_mode` | literal | `chat`, `grounded_answer`, `summary`, `comparison`, `action`, `fallback` |
| `action_name` | string \| null | set only for `task_action` |

## Errors

All endpoints wrap internal exceptions as HTTP 500 with `{"detail": "<message>"}`. Pydantic validation errors return 422 as usual.

## Ingest

HoloChat does **not** expose ingest endpoints. Use the existing Hologram endpoints:

- `POST /ingest/file` on `hologram/server.py` (port 8000 by default) for PDF/DOCX
- Or call `Hologram.ingest_file(glyph_id, path)` directly in Python

The chatbot reads from whichever Hologram project its requests specify via `project`.
