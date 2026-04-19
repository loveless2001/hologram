"""Shared schemas for the holochat router and API."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


IntentLiteral = Literal[
    "chat",
    "knowledge_qa",
    "doc_qa",
    "summarization",
    "comparison",
    "session_memory",
    "task_action",
    "fallback",
]
RetrievalScopeLiteral = Literal[
    "none",
    "global_corpus",
    "active_document",
    "selected_documents",
    "session_memory",
]
ResponseModeLiteral = Literal[
    "chat",
    "grounded_answer",
    "summary",
    "comparison",
    "action",
    "fallback",
]


class RouteDecision(BaseModel):
    intent: IntentLiteral
    confidence: float
    needs_retrieval: bool
    retrieval_scope: RetrievalScopeLiteral = "none"
    needs_rerank: bool = False
    needs_chat_history: bool = False
    response_mode: ResponseModeLiteral = "fallback"
    action_name: Optional[str] = None


class RouterContext(BaseModel):
    user_message: str
    active_document_id: Optional[str] = None
    selected_document_ids: List[str] = Field(default_factory=list)
    recent_messages: List[str] = Field(default_factory=list)
    session_summary: Optional[str] = None
    just_uploaded_files: bool = False
    available_actions: List[str] = Field(default_factory=list)
    corpus_search_enabled: bool = True


class RouteRequest(BaseModel):
    project: str
    session_id: Optional[str] = None
    context: RouterContext


class Citation(BaseModel):
    trace_id: str
    score: float
    content: str
    source_doc: Optional[str] = None
    page_number: Optional[int] = None
    meta: Dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    route: RouteDecision


class ChatRequest(BaseModel):
    project: str
    session_id: str
    context: RouterContext
    top_k: int = 5


class ChatResponse(BaseModel):
    session_id: str
    route: RouteDecision
    reply: str
    citations: List[Citation] = Field(default_factory=list)
