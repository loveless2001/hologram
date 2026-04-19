"""Merge rule-based and prototype-based routing into a final decision."""

from __future__ import annotations

from typing import Optional

from ..schemas import RouteDecision, RouterContext
from .classifier import PrototypeIntentClassifier
from .llm_router import LLMRouter
from .rules import apply_rules, has_unresolved_reference, preferred_document_scope


def _decision_for_intent(
    intent: str,
    ctx: RouterContext,
    confidence: float,
) -> RouteDecision:
    if intent == "chat":
        return RouteDecision(
            intent="chat",
            confidence=confidence,
            needs_retrieval=False,
            retrieval_scope="none",
            needs_chat_history=True,
            response_mode="chat",
        )
    if intent == "knowledge_qa":
        return RouteDecision(
            intent="knowledge_qa",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope="global_corpus",
            needs_rerank=True,
            response_mode="grounded_answer",
        )
    if intent == "doc_qa":
        return RouteDecision(
            intent="doc_qa",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope=preferred_document_scope(ctx),
            needs_rerank=True,
            response_mode="grounded_answer",
        )
    if intent == "summarization":
        return RouteDecision(
            intent="summarization",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope=preferred_document_scope(ctx),
            needs_rerank=True,
            response_mode="summary",
        )
    if intent == "comparison":
        scope = "selected_documents" if len(ctx.selected_document_ids) >= 2 else "global_corpus"
        return RouteDecision(
            intent="comparison",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope=scope,
            needs_rerank=True,
            response_mode="comparison",
        )
    if intent == "session_memory":
        return RouteDecision(
            intent="session_memory",
            confidence=confidence,
            needs_retrieval=True,
            retrieval_scope="session_memory",
            needs_chat_history=True,
            response_mode="grounded_answer",
        )
    if intent == "task_action":
        return RouteDecision(
            intent="task_action",
            confidence=confidence,
            needs_retrieval=False,
            retrieval_scope="none",
            response_mode="action",
        )
    return RouteDecision(
        intent="fallback",
        confidence=confidence,
        needs_retrieval=False,
        retrieval_scope="none",
        response_mode="fallback",
    )


def _merge_decisions(
    rule_decision: Optional[RouteDecision],
    classifier_decision: RouteDecision,
    ctx: RouterContext,
) -> RouteDecision:
    def _copy(decision: RouteDecision, **updates) -> RouteDecision:
        if hasattr(decision, "model_copy"):
            return decision.model_copy(update=updates)
        return decision.copy(update=updates)

    if rule_decision is None:
        merged = classifier_decision
    elif rule_decision.intent == classifier_decision.intent:
        boosted = min(0.99, max(rule_decision.confidence, classifier_decision.confidence + 0.15))
        merged = _copy(classifier_decision, confidence=boosted)
    elif rule_decision.confidence >= classifier_decision.confidence + 0.15:
        merged = rule_decision
    else:
        lowered = max(0.1, classifier_decision.confidence * 0.75)
        merged = _copy(classifier_decision, confidence=lowered)

    if has_unresolved_reference(ctx.user_message) and not (
        ctx.active_document_id or ctx.selected_document_ids
    ):
        if merged.intent in {"doc_qa", "summarization", "comparison"}:
            return RouteDecision(
                intent="fallback",
                confidence=min(merged.confidence, 0.45),
                needs_retrieval=False,
                retrieval_scope="none",
                response_mode="fallback",
            )

    return merged


def decide_route(
    ctx: RouterContext,
    classifier: PrototypeIntentClassifier,
    llm_router: Optional[LLMRouter] = None,
) -> RouteDecision:
    rule_decision = apply_rules(ctx)
    if rule_decision is not None and rule_decision.confidence >= 0.9:
        return rule_decision

    classification = classifier.classify(ctx.user_message)
    classifier_decision = _decision_for_intent(
        classification.intent,
        ctx,
        classification.confidence,
    )
    merged = _merge_decisions(rule_decision, classifier_decision, ctx)

    if merged.confidence < 0.6 and llm_router is not None:
        llm_decision = llm_router.route(ctx)
        if llm_decision is not None:
            return llm_decision

    if merged.confidence < 0.45:
        return RouteDecision(
            intent="fallback",
            confidence=merged.confidence,
            needs_retrieval=False,
            retrieval_scope="none",
            response_mode="fallback",
        )

    return merged
