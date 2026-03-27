from asyncio import Future

from dataclasses import dataclass
from typing import Any, AsyncIterator, Sequence

from core.chat import ChatMessage
from core.event_loop import Event, InputEvent
from core.event_loop.single_event_processor import SingleEventProcessor
from core.event_processors.message import DeferredToolCallBatch, MessageEvent


@dataclass
class SubAgentEvent(InputEvent):
    handle: Future
    data: Sequence[ChatMessage | DeferredToolCallBatch]

class SubagentEventProcessor(SingleEventProcessor[SubAgentEvent, Event]):
    def __init__(self, message_processor: SingleEventProcessor) -> None:
        self.message_processor = message_processor
    
    async def process(self, event: SubAgentEvent) -> AsyncIterator[Event]:
        event.handle.set_result("Sub-agent completed instruction")  # Simulate sub-agent work and completion
        async for e in self.message_processor.process(event):
            yield e

