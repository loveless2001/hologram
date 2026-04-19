"""Local session state for chat history and lightweight memory lookup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from .schemas import Citation


@dataclass
class SessionMessage:
    role: str
    content: str


class InMemorySessionStore:
    def __init__(self) -> None:
        self._messages: Dict[str, List[SessionMessage]] = {}
        self._summaries: Dict[str, str] = {}

    def append_message(self, session_id: str, role: str, content: str) -> None:
        self._messages.setdefault(session_id, []).append(SessionMessage(role=role, content=content))

    def get_recent_messages(self, session_id: str, *, limit: int = 3) -> List[str]:
        messages = self._messages.get(session_id, [])
        return [msg.content for msg in messages[-limit:]]

    def get_summary(self, session_id: str) -> Optional[str]:
        return self._summaries.get(session_id)

    def set_summary(self, session_id: str, summary: str) -> None:
        self._summaries[session_id] = summary

    def search_messages(self, session_id: str, query: str, *, limit: int = 3) -> List[Citation]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        hits = []
        for idx, msg in enumerate(self._messages.get(session_id, [])):
            overlap = len(query_tokens.intersection(self._tokenize(msg.content)))
            if overlap <= 0:
                continue
            score = overlap / max(1, len(query_tokens))
            hits.append(
                Citation(
                    trace_id=f"session:{session_id}:{idx}",
                    score=float(score),
                    content=f"{msg.role}: {msg.content}",
                    meta={"session_id": session_id, "role": msg.role},
                )
            )
        hits.sort(key=lambda item: item.score, reverse=True)
        return hits[:limit]

    @staticmethod
    def _tokenize(text: str) -> Set[str]:
        return {token.strip(".,!?").lower() for token in text.split() if token.strip(".,!?")}
