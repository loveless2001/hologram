"""Optional LLM router boundary for ambiguous cases."""

from __future__ import annotations

from typing import Optional, Protocol

from ..schemas import RouteDecision, RouterContext


class LLMRouter(Protocol):
    def route(self, ctx: RouterContext) -> Optional[RouteDecision]:
        """Return a route decision for ambiguous requests, or None."""


class DisabledLLMRouter:
    def route(self, ctx: RouterContext) -> Optional[RouteDecision]:
        return None
