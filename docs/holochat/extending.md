# Extending HoloChat

Short recipes for the most common v1-era changes. For bigger structural work, open a plan under `plans/` first.

## Add a new intent

1. Widen the `Intent` literal in `apps/holochat/schemas.py`.
2. Add a branch in `router/decision.py::_decision_for_intent` that returns a `RouteDecision` with the right scope + response_mode.
3. Add seed utterances in `router/prototypes.py` (10–30 examples) so the classifier can recognize it.
4. If the intent has deterministic triggers, add a rule in `router/rules.py`.
5. Add orchestrator behavior in `apps/holochat/orchestrator.py::_deterministic_reply` (and optionally `_system_prompt` if the LLM path needs different guidance).
6. Unit-test the rule + classifier + merge logic under `apps/holochat/router/tests/`.
7. Integration-test the new intent in `tests/test_holochat_integration_end_to_end.py`.

## Add a new retrieval scope

1. Widen `RetrievalScope` literal in `schemas.py`.
2. Extend `retrieval_adapter.py::_doc_ids_for_scope` (or add a new branch) to translate the scope into filters for `Hologram.search_scoped(...)`.
3. If the scope needs filters that don't map to `doc_ids`/`glyph_ids`, pass a `trace_filter` callable instead.
4. Document the scope in `docs/holochat/router-intents.md`.

## Add a new provider

1. Implement the `ChatProvider` protocol in `apps/holochat/providers.py`:
   ```python
   class MyProvider:
       def generate(self, messages: List[Dict[str, str]]) -> str: ...
   ```
2. Extend `build_provider()` to try your provider before/after OpenAI based on env vars.
3. No orchestrator changes needed — it consumes `ChatProvider` through the protocol.

## Add a new task_action

1. Add a regex in `router/rules.py::ACTION_PATTERNS` keyed by the action name.
2. Add an executor — v1 has no action registry, so for now handle the action name in `orchestrator.py::_deterministic_reply` or wire a `task_action` dispatcher before reply generation.
3. Document the action name in `api-reference.md`.

## Swap in a real reranker

Reranking is stubbed in v1 (no-op). When you wire a real one:

1. Add a `rerank_traces(traces, query) -> traces` function, e.g. in a new `apps/holochat/rerank.py`.
2. Call it inside the orchestrator between retrieval and reply generation, guarded by `decision.needs_rerank`.
3. Add a feature flag so it can be toggled off during benchmarks.

## Promote session store from in-memory to SQLite

The in-memory store in `apps/holochat/session_store.py` implements a small interface. To swap:

1. Create `apps/holochat/sqlite_session_store.py` with the same methods: `append_message`, `get_recent_messages`, `get_summary`, `set_summary`, `search_messages`.
2. Inject it at orchestrator construction time: `ChatOrchestrator(session_store=SqliteSessionStore(...))`.
3. No other module needs to change.

Schema sketch lives in `plans/260420-0213-holochat-v1/phase-04-api-endpoints-and-session-store.md`.

## Add vector search over session memory

v1 uses keyword overlap in `InMemorySessionStore.search_messages`. To upgrade:

1. When appending a message, also ingest it into Hologram under a per-session glyph (e.g. `session:{session_id}`).
2. In `orchestrator.respond` for `intent == "session_memory"`, call `hologram.search_scoped(query, glyph_ids=[f"session:{session_id}"])` instead of the keyword path.
3. Gate behind a config flag until quality is measured.

## Debugging tips

- `/chat/route` returns the classifier decision without running the pipeline — use it to isolate routing bugs from retrieval/reply bugs.
- Set `OPENAI_API_KEY=""` (or unset it) to force the deterministic reply path — makes retrieval/routing bugs easier to spot.
- `Hologram.search_scoped(query, doc_ids=[...], mode="global")` in a REPL is the fastest way to validate that a doc's traces are indexed with the right `source_doc`.
