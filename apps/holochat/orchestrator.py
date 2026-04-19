"""Route requests, fetch context, and produce a chatbot reply."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

from hologram.api import Hologram

from .providers import ChatProvider
from .retrieval_adapter import HologramRetrievalAdapter
from .router.classifier import PrototypeIntentClassifier
from .router.decision import decide_route
from .schemas import ChatResponse, Citation, RouteDecision, RouterContext
from .session_store import InMemorySessionStore


class ChatOrchestrator:
    def __init__(
        self,
        *,
        session_store: Optional[InMemorySessionStore] = None,
        retrieval_adapter: Optional[HologramRetrievalAdapter] = None,
        provider: Optional[ChatProvider] = None,
    ) -> None:
        self.session_store = session_store or InMemorySessionStore()
        self.retrieval_adapter = retrieval_adapter or HologramRetrievalAdapter()
        self.provider = provider

    def _hydrate_context(self, session_id: str, ctx: RouterContext) -> RouterContext:
        recent_messages = list(ctx.recent_messages)
        if not recent_messages:
            recent_messages = self.session_store.get_recent_messages(session_id, limit=3)
        session_summary = ctx.session_summary or self.session_store.get_summary(session_id)
        payload = ctx.model_dump() if hasattr(ctx, "model_dump") else ctx.dict()
        return RouterContext(
            **{
                **payload,
                "recent_messages": recent_messages,
                "session_summary": session_summary,
            }
        )

    def route(self, hologram: Hologram, ctx: RouterContext) -> RouteDecision:
        classifier = PrototypeIntentClassifier(
            lambda text: hologram.manifold.align_text(text, hologram.text_encoder)
        )
        return decide_route(ctx, classifier)

    def respond(
        self,
        hologram: Hologram,
        session_id: str,
        ctx: RouterContext,
        *,
        top_k: int = 5,
    ) -> ChatResponse:
        hydrated = self._hydrate_context(session_id, ctx)
        decision = self.route(hologram, hydrated)

        citations: List[Citation] = []
        if decision.intent == "session_memory":
            citations = self.session_store.search_messages(
                session_id,
                hydrated.user_message,
                limit=top_k,
            )
        elif decision.needs_retrieval:
            citations = self.retrieval_adapter.search(
                hologram,
                hydrated,
                decision,
                top_k=top_k,
            )

        reply = self._generate_reply(hydrated, decision, citations)

        self.session_store.append_message(session_id, "user", hydrated.user_message)
        self.session_store.append_message(session_id, "assistant", reply)

        return ChatResponse(
            session_id=session_id,
            route=decision,
            reply=reply,
            citations=citations,
        )

    def _generate_reply(
        self,
        ctx: RouterContext,
        decision: RouteDecision,
        citations: List[Citation],
    ) -> str:
        if self.provider is not None:
            llm_reply = self._llm_reply(ctx, decision, citations)
            if llm_reply:
                return llm_reply
        return self._deterministic_reply(ctx, decision, citations)

    def _llm_reply(
        self,
        ctx: RouterContext,
        decision: RouteDecision,
        citations: List[Citation],
    ) -> str:
        messages = [{"role": "system", "content": self._system_prompt(decision)}]
        for message in ctx.recent_messages[-3:]:
            messages.append({"role": "user", "content": message})
        if citations:
            messages.append(
                {
                    "role": "system",
                    "content": self._citation_block(citations),
                }
            )
        messages.append({"role": "user", "content": ctx.user_message})
        return self.provider.generate(messages)

    @staticmethod
    def _system_prompt(decision: RouteDecision) -> str:
        if decision.response_mode == "summary":
            return "Summarize only from the supplied context. Be concise and grounded."
        if decision.response_mode == "comparison":
            return "Compare the supplied context. Be explicit about which source supports each point."
        if decision.response_mode == "action":
            return "Acknowledge the requested action and explain the current execution status."
        return "Answer using the supplied context when present. Do not invent citations."

    @staticmethod
    def _citation_block(citations: List[Citation]) -> str:
        lines = ["Grounding context:"]
        for citation in citations:
            source_doc = citation.source_doc or "session"
            lines.append(f"- [{source_doc}] {citation.content}")
        return "\n".join(lines)

    def _deterministic_reply(
        self,
        ctx: RouterContext,
        decision: RouteDecision,
        citations: List[Citation],
    ) -> str:
        if decision.intent == "chat":
            return "I can help with document Q&A, summaries, comparisons, and session recall."
        if decision.intent == "task_action":
            return f"Routed as action `{decision.action_name or 'unspecified'}`. Execution is not wired yet."
        if decision.intent == "fallback":
            return "I need a more specific request or a document selection to route this safely."
        if not citations:
            return "I could not find grounded context for that request."
        if decision.intent == "summarization":
            snippets = [citation.content.strip() for citation in citations[:3]]
            return "Summary: " + " ".join(snippets)
        if decision.intent == "comparison":
            grouped: Dict[str, List[str]] = defaultdict(list)
            for citation in citations:
                grouped[citation.source_doc or "unknown"].append(citation.content.strip())
            parts = []
            for source_doc, snippets in grouped.items():
                parts.append(f"{source_doc}: {' '.join(snippets[:2])}")
            return "Comparison: " + " | ".join(parts)
        if decision.intent == "session_memory":
            return "Session memory: " + " ".join(citation.content for citation in citations[:3])
        return "Grounded answer: " + " ".join(citation.content.strip() for citation in citations[:3])
