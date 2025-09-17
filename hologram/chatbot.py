"""High-level chat orchestration with holographic memory.

This module wires the `Hologram` store into a lightweight chat workflow. It
exposes:

* Provider abstractions (`ChatProvider`, `OpenAIChatProvider`, `EchoProvider`).
* `SessionLog` to persist plain-text chat logs per session.
* `ChatMemory` to record and retrieve messages in holographic memory.
* `ChatWindow` offering a CLI-friendly loop that keeps sessions alive and
  cross-references prior sessions via semantic search.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol

from .api import Hologram
from .store import Trace


# ---------------------------------------------------------------------------
# Provider abstractions
# ---------------------------------------------------------------------------
class ChatProvider(Protocol):
    """Minimal protocol for LLM chat providers."""

    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Return the assistant response given an OpenAI-style message list."""


class OpenAIChatProvider:
    """Thin wrapper around the official OpenAI client."""

    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised when missing dep
            raise RuntimeError(
                "openai package is required for OpenAIChatProvider. Install it with"
                " `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return completion.choices[0].message.content.strip()


class EchoProvider:
    """Fallback provider that simply echoes the last user prompt."""

    def generate(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover - tiny
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return f"(echo) {msg.get('content', '')}"
        return "(echo)"


# ---------------------------------------------------------------------------
# Session logging helpers
# ---------------------------------------------------------------------------
@dataclass
class SessionLog:
    """Append-only JSONL logs per session for easy comparison."""

    root: Path

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.root / f"{session_id}.jsonl"

    def append(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        timestamp: Optional[float] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        payload = {
            "role": role,
            "content": content,
            "timestamp": timestamp if timestamp is not None else time.time(),
        }
        if extra:
            payload.update(extra)

        path = self._path(session_id)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def read(self, session_id: str) -> List[Dict]:
        path = self._path(session_id)
        if not path.exists():
            return []
        lines = []
        with path.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                lines.append(json.loads(line))
        return lines


# ---------------------------------------------------------------------------
# Holographic memory helpers
# ---------------------------------------------------------------------------
@dataclass
class ChatMemory:
    hologram: Hologram
    session_prefix: str = "session"
    global_glyph: str = "chat:global"
    session_window: int = 6
    cross_session_k: int = 3

    def session_glyph(self, session_id: str) -> str:
        return f"chat:{self.session_prefix}:{session_id}"

    def record_message(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        timestamp: Optional[float] = None,
        trace_id: Optional[str] = None,
    ) -> str:
        """Store the message in session + global glyphs."""

        trace_id = trace_id or f"chat:{uuid.uuid4().hex}"
        ts = timestamp if timestamp is not None else time.time()
        meta = {"role": role, "session_id": session_id, "timestamp": ts}

        vec = self.hologram.text_encoder.encode(content)
        trace = Trace(trace_id=trace_id, kind="chat", content=content, vec=vec, meta=meta)
        self.hologram.glyphs.attach_trace(self.session_glyph(session_id), trace)
        # Link the same trace to the global glyph for cross-session comparisons.
        self.hologram.store.link_trace(self.global_glyph, trace_id)
        return trace_id

    def get_recent_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        traces = self.hologram.recall_glyph(self.session_glyph(session_id))
        traces = [t for t in traces if t is not None]
        traces.sort(key=lambda tr: tr.meta.get("timestamp", 0.0))
        recent = traces[-self.session_window :]
        return [
            {"role": tr.meta.get("role", "assistant"), "content": tr.content}
            for tr in recent
        ]

    def search_global_context(
        self,
        query: str,
        *,
        exclude_session: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        limit = top_k or self.cross_session_k
        hits = self.hologram.search_text(query, top_k=limit * 3)
        out: List[Dict] = []
        for trace, score in hits:
            session_id = trace.meta.get("session_id")
            if exclude_session and session_id == exclude_session:
                continue
            out.append(
                {
                    "role": trace.meta.get("role", "assistant"),
                    "session_id": session_id,
                    "content": trace.content,
                    "score": score,
                }
            )
            if len(out) >= limit:
                break
        return out

    def build_cross_session_prompt(self, session_id: str, query: str) -> Optional[str]:
        hits = self.search_global_context(query, exclude_session=session_id)
        if not hits:
            return None
        lines = ["Relevant memories from other sessions:"]
        for hit in hits:
            session_lbl = hit.get("session_id", "?")
            role = hit.get("role", "assistant")
            lines.append(f"- ({session_lbl}, {role}) {hit.get('content', '')}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------
@dataclass
class ChatWindow:
    provider: ChatProvider
    memory: ChatMemory
    logs: SessionLog
    session_id: str
    system_prompt: Optional[str] = None

    def _base_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        return messages

    def step(self, user_text: str) -> str:
        """Process a single user turn and return the assistant reply."""

        context_msgs = self.memory.get_recent_session_messages(self.session_id)
        cross_prompt = self.memory.build_cross_session_prompt(self.session_id, user_text)

        messages = self._base_messages()
        if cross_prompt:
            messages.append({"role": "system", "content": cross_prompt})
        messages.extend(context_msgs)
        messages.append({"role": "user", "content": user_text})

        reply = self.provider.generate(messages)

        ts = time.time()
        self.memory.record_message(self.session_id, "user", user_text, timestamp=ts)
        self.logs.append(self.session_id, "user", user_text, timestamp=ts)

        reply_ts = time.time()
        self.memory.record_message(self.session_id, "assistant", reply, timestamp=reply_ts)
        self.logs.append(self.session_id, "assistant", reply, timestamp=reply_ts)

        return reply

    # Simple CLI driver -----------------------------------------------------
    def run_cli(self) -> None:
        print(f"Starting session '{self.session_id}'. Type /exit to finish.")
        existing = self.logs.read(self.session_id)
        if existing:
            print(f"Loaded {len(existing)} previous log entries for this session.")
        try:
            while True:
                try:
                    user_text = input("you> ").strip()
                except EOFError:
                    print()
                    break

                if not user_text:
                    continue
                if user_text.lower() in {"/exit", "/quit"}:
                    break

                reply = self.step(user_text)
                print(f"bot> {reply}\n")
        except KeyboardInterrupt:
            print("\nSession interrupted.")


def resolve_provider(api_key: Optional[str] = None, model: str = "gpt-3.5-turbo") -> ChatProvider:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if key:
        try:
            return OpenAIChatProvider(api_key=key, model=model)
        except RuntimeError as exc:
            print(f"[warn] Falling back to echo provider: {exc}")
    else:
        print("[warn] OPENAI_API_KEY not set – using echo provider.")
    return EchoProvider()


__all__ = [
    "ChatProvider",
    "OpenAIChatProvider",
    "EchoProvider",
    "ChatMemory",
    "SessionLog",
    "ChatWindow",
    "resolve_provider",
]
