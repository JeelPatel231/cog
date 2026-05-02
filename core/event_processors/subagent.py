from dataclasses import dataclass
from typing import TypeGuard, override
from collections.abc import AsyncIterator

from core.event_loop import Event, OutputEvent
from core.event_loop.single_event_processor import SingleEventProcessor


@dataclass
class SubAgentThinkingOutput(OutputEvent):
    data: str

class SubAgentThinkingEventProcessor(SingleEventProcessor[SubAgentThinkingOutput, Event]):
    
    @override
    async def can_process(self, event: Event) -> TypeGuard[SubAgentThinkingOutput]:
        return isinstance(event, SubAgentThinkingOutput)

    @override
    async def process(self, event: SubAgentThinkingOutput) -> AsyncIterator[Event]:
        print(f"SubAgent Output: {event.data}")
        return
        yield