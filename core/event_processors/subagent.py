from dataclasses import dataclass
from typing import AsyncIterator, TypeGuard

from core.event_loop import Event, OutputEvent
from core.event_loop.single_event_processor import SingleEventProcessor


@dataclass
class SubAgentThinkingOutput(OutputEvent):
    data: str

class SubAgentThinkingEventProcessor(SingleEventProcessor[SubAgentThinkingOutput, Event]):
    async def can_process(self, event: Event) -> TypeGuard[SubAgentThinkingOutput]:
        return isinstance(event, SubAgentThinkingOutput)

    async def process(self, event: SubAgentThinkingOutput) -> AsyncIterator[Event]:
        print(f"SubAgent Output: {event.data}")
        return
        yield