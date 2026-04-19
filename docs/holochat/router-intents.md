# Router Intents

8 intents, 5 retrieval scopes, 6 response modes. Each user turn resolves to exactly one `(intent, scope, response_mode)` triple via a two-layer router: deterministic rules → embedding prototype classifier. A third LLM-router layer is stubbed and off by default in v1.

## Intents

| Intent | Triggers (examples) | Retrieval | Scope | Response mode |
|---|---|---|---|---|
| `chat` | "hi", "thanks", "what do you think" | no | none | `chat` |
| `knowledge_qa` | "what is ETOPS", "explain X" | yes | `global_corpus` | `grounded_answer` |
| `doc_qa` | "what does this doc say", "summarize section 3" (with active doc) | yes | `active_document` or `selected_documents` | `grounded_answer` |
| `summarization` | "summarize this", "key points", "tl;dr" | yes | `active_document` \| `selected_documents` \| `global_corpus` | `summary` |
| `comparison` | "compare these", "difference between v1 and v2" | yes | `selected_documents` if ≥2, else `global_corpus` | `comparison` |
| `session_memory` | "what did we discuss", "remind me" | yes | `session_memory` | `grounded_answer` |
| `task_action` | "index this", "re-embed", "delete" | no | none | `action` (with `action_name`) |
| `fallback` | low confidence, unresolved references with no doc context | no | none | `fallback` |

## Scope → retrieval behavior

| Scope | Source |
|---|---|
| `none` | no retrieval |
| `global_corpus` | `Hologram.search_scoped(mode="dynamic")` over the whole project |
| `active_document` | `search_scoped(doc_ids=[ctx.active_document_id])` |
| `selected_documents` | `search_scoped(doc_ids=ctx.selected_document_ids)` |
| `session_memory` | `SessionStore.search_messages(session_id, query)` (keyword overlap, not vector search in v1) |

## Rule layer (layer 1)

High-confidence deterministic handling. Patterns live in `apps/holochat/router/rules.py`:

- Greetings/thanks regex → `chat` (confidence 0.97)
- Session memory phrases ("what did we decide", "remind me") → `session_memory` (0.96)
- Summarize keywords → `summarization` with scope from `preferred_document_scope(ctx)`
- Compare keywords with ≥2 selected docs → `comparison`
- Action verbs (index/ingest/upload/delete/re-embed/pin) → `task_action` with populated `action_name`
- Pronoun-heavy follow-ups without document context → downgrade to `fallback`

If a rule fires with confidence ≥ 0.9, it short-circuits the classifier. Otherwise the classifier runs and `decision.py` merges both.

## Classifier layer (layer 2)

`apps/holochat/router/classifier.py` — embedding prototype classifier:

- Seed utterances per intent live in `router/prototypes.py`
- Encoded once at startup via `Hologram.manifold.align_text(...)` (same MiniLM the engine uses)
- Per-query: embed user message → cosine similarity to each intent centroid → top-1 intent + confidence from top1/top2 margin
- Confidence bands:
  - `≥ 0.80` high — use directly
  - `0.60–0.79` medium — use, orchestrator may broaden retrieval
  - `< 0.45` low — coerce to `fallback`

## Merge logic

`router/decision.py::_merge_decisions`:

- If rule and classifier agree → boost confidence
- If rule is strictly higher confidence (+0.15) → rule wins
- Otherwise → classifier wins, confidence damped (×0.75)
- Pronoun override: if message has unresolved references AND no active/selected docs AND intent is one of `doc_qa`/`summarization`/`comparison` → force `fallback`

## LLM fallback (stubbed, off)

`router/llm_router.py` provides an interface `LLMRouter.route(ctx) -> Optional[RouteDecision]`. In v1 no implementation is registered; `decide_route` skips this step. V2 slots in a real implementation behind a confidence threshold (default `< 0.6`).

## Testing

- Codex's router unit tests live under `apps/holochat/router/tests/` (test each layer in isolation)
- End-to-end intent behavior validated in `tests/test_holochat_integration_end_to_end.py`
