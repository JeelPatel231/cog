from typing import Any, Protocol, Sequence

from core.chat import ChatMessage, SystemMessage, TextMessageContent
from core.skills import SkillRegistry


class HistoryTransformer(Protocol):
    """
    A pure transformation step that runs before the AI API call.

    Takes the full application-level chat history and returns the list of
    messages that should actually be sent to the model. Implementations can
    strip internal-only messages, compact context, truncate long histories,
    inject synthetic context, etc.

    The original history must never be mutated — always return a new list.
    """

    def transform(self, history: Sequence[Any]) -> list[ChatMessage]: ...


class PassthroughHistoryTransformer:
    """Default no-op transformer — passes the full history through unchanged."""

    def transform(self, history: Sequence[ChatMessage]) -> list[ChatMessage]:
        return list(history)

