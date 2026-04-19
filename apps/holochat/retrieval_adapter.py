"""Thin adapter between holochat decisions and Hologram retrieval."""

from __future__ import annotations

from typing import List, Optional

from hologram.api import Hologram

from .schemas import Citation, RouteDecision, RouterContext


class HologramRetrievalAdapter:
    def _doc_ids_for_scope(
        self,
        ctx: RouterContext,
        decision: RouteDecision,
    ) -> Optional[List[str]]:
        if decision.retrieval_scope == "active_document":
            return [ctx.active_document_id] if ctx.active_document_id else []
        if decision.retrieval_scope == "selected_documents":
            return list(ctx.selected_document_ids)
        return None

    def search(
        self,
        hologram: Hologram,
        ctx: RouterContext,
        decision: RouteDecision,
        *,
        top_k: int = 5,
    ) -> List[Citation]:
        doc_ids = self._doc_ids_for_scope(ctx, decision)
        if doc_ids == []:
            return []

        traces = hologram.search_scoped(
            ctx.user_message,
            top_k=top_k,
            doc_ids=doc_ids,
            mode="dynamic",
        )
        return [
            Citation(
                trace_id=trace.trace_id,
                score=score,
                content=trace.content,
                source_doc=trace.meta.get("source_doc"),
                page_number=trace.meta.get("page_number"),
                meta=dict(trace.meta),
            )
            for trace, score in traces
        ]
