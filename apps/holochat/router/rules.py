"""Deterministic routing rules for high-confidence requests."""

from __future__ import annotations

import re
from typing import Optional

from ..schemas import RouteDecision, RouterContext


REFERENCE_RE = re.compile(r"\b(this|that|it|these|those)\b", re.IGNORECASE)
GREETING_RE = re.compile(r"^\s*(hi|hello|hey|thanks|thank you)\s*[!.?]*\s*$", re.IGNORECASE)
SUMMARY_RE = re.compile(r"\b(summarize|summary|key points|tl\s*;?\s*dr|condense)\b", re.IGNORECASE)
COMPARE_RE = re.compile(r"\b(compare|difference between|what changed)\b", re.IGNORECASE)
SESSION_MEMORY_RE = re.compile(
    r"\b(what did we (say|decide|discuss)|continue from|what was the plan|remind me)\b",
    re.IGNORECASE,
)
DOC_RE = re.compile(
    r"\b(this|that|the)\s+(doc|document|file|section|page)\b|\bsection\s+\d+\b",
    re.IGNORECASE,
)
ACTION_PATTERNS = {
    "index_file": re.compile(r"\b(index|ingest|upload)\b", re.IGNORECASE),
    "delete_source": re.compile(r"\bdelete\b", re.IGNORECASE),
    "reembed_corpus": re.compile(r"\b(re[- ]?embed|reindex)\b", re.IGNORECASE),
    "pin_answer": re.compile(r"\bpin\b", re.IGNORECASE),
}


def has_unresolved_reference(text: str) -> bool:
    return bool(REFERENCE_RE.search(text))


def preferred_document_scope(ctx: RouterContext) -> str:
    if ctx.selected_document_ids:
        return "selected_documents"
    if ctx.active_document_id:
        return "active_document"
    return "global_corpus"


def _action_name(text: str) -> Optional[str]:
    for action_name, pattern in ACTION_PATTERNS.items():
        if pattern.search(text):
            return action_name
    return None


def apply_rules(ctx: RouterContext) -> Optional[RouteDecision]:
    text = ctx.user_message.strip()
    if not text:
        return RouteDecision(
            intent="fallback",
            confidence=0.0,
            needs_retrieval=False,
            retrieval_scope="none",
            response_mode="fallback",
        )

    if GREETING_RE.match(text):
        return RouteDecision(
            intent="chat",
            confidence=0.97,
            needs_retrieval=False,
            retrieval_scope="none",
            needs_chat_history=True,
            response_mode="chat",
        )

    if SESSION_MEMORY_RE.search(text):
        return RouteDecision(
            intent="session_memory",
            confidence=0.96,
            needs_retrieval=True,
            retrieval_scope="session_memory",
            needs_chat_history=True,
            response_mode="grounded_answer",
        )

    action_name = _action_name(text)
    if action_name is not None:
        return RouteDecision(
            intent="task_action",
            confidence=0.94,
            needs_retrieval=False,
            retrieval_scope="none",
            response_mode="action",
            action_name=action_name,
        )

    if COMPARE_RE.search(text):
        if len(ctx.selected_document_ids) >= 2:
            return RouteDecision(
                intent="comparison",
                confidence=0.97,
                needs_retrieval=True,
                retrieval_scope="selected_documents",
                needs_rerank=True,
                response_mode="comparison",
            )
        return RouteDecision(
            intent="fallback",
            confidence=0.4,
            needs_retrieval=False,
            retrieval_scope="none",
            response_mode="fallback",
        )

    if SUMMARY_RE.search(text):
        scope = preferred_document_scope(ctx)
        confidence = 0.96 if scope != "global_corpus" or ctx.just_uploaded_files else 0.72
        return RouteDecision(
            intent="summarization",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope=scope,
            needs_rerank=True,
            response_mode="summary",
        )

    if DOC_RE.search(text) or (
        has_unresolved_reference(text)
        and (ctx.active_document_id is not None or bool(ctx.selected_document_ids))
    ):
        scope = preferred_document_scope(ctx)
        return RouteDecision(
            intent="doc_qa",
            confidence=0.92 if scope != "global_corpus" else 0.55,
            needs_retrieval=True,
            retrieval_scope=scope,
            needs_rerank=True,
            response_mode="grounded_answer",
        )

    if ctx.corpus_search_enabled and re.search(r"\b(what|how|why|explain)\b", text, re.IGNORECASE):
        return RouteDecision(
            intent="knowledge_qa",
            confidence=0.74,
            needs_retrieval=True,
            retrieval_scope="global_corpus",
            needs_rerank=True,
            response_mode="grounded_answer",
        )

    return None
