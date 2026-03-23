from typing import Protocol

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

    def transform(self, history: list[ChatMessage]) -> list[ChatMessage]: ...


class PassthroughHistoryTransformer:
    """Default no-op transformer — passes the full history through unchanged."""

    def transform(self, history: list[ChatMessage]) -> list[ChatMessage]:
        return list(history)


class SkillMetadataTransformer:
    """
    Prepends a system message listing the name and description of every
    registered skill. This is the Level-1 (metadata-only) load: Claude learns
    all available skills at startup without pulling in their full instructions.

    The full SKILL.md for any skill is loaded on demand when Claude calls
    the 'load_skill' tool — which may happen once or many times per turn,
    once for each skill Claude decides it needs.
    """

    def __init__(self, skill_registry: SkillRegistry) -> None:
        self._registry = skill_registry

    def transform(self, history: list[ChatMessage]) -> list[ChatMessage]:
        skills = self._registry.list_skills()
        if not skills:
            return list(history)

        lines = [
            "You have access to the following skills.",
            "Use the 'load_skill' tool to load a skill's full instructions whenever a task matches its description.",
            "You may load as many skills as the task requires.\n",
        ]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}")

        system_msg = SystemMessage(
            role="system",
            content=TextMessageContent(text="\n".join(lines)),
        )
        return [system_msg, *history]
