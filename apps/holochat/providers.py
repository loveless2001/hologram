"""Optional provider integrations for holochat."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Protocol


class ChatProvider(Protocol):
    def generate(self, messages: List[Dict[str, str]]) -> str:
        """Return an assistant message for the given chat payload."""


class OpenAIChatProvider:
    def __init__(self, api_key: str, model: str = "gpt-5") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for OpenAIChatProvider") from exc

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, messages: List[Dict[str, str]]) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return completion.choices[0].message.content.strip()


def build_provider(model: str = "gpt-5") -> Optional[ChatProvider]:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover
        load_dotenv = None

    if load_dotenv is not None:
        load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        return OpenAIChatProvider(api_key=api_key, model=model)
    except RuntimeError:
        return None
