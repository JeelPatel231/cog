from core.chat import ChatProtocol
from core.tool_provider import ToolProvider
from core.event_loop.processor_registry import EventProcessorRegistry
from . import Event

from .loop import EventLoop

class EventLoopProcessor:
    def __init__(
        self,
        event_loop: EventLoop,
        event_processor_registry: EventProcessorRegistry,
    ):
        self.event_loop = event_loop
        self.event_processors = event_processor_registry

    async def process(self, event: Event):
        """
        take an event from the loop. the event will be a serialized conversation containing user and assistant messages.
        the set of messages will be passed to the agent, which will return a response. the response will be appended to the conversation.
        the new conversation will be serialized and put back on the event loop for the next processor to handle.

        the updated conversation should only be put back in the event loop if there are tool calls being made.
        """
        event_type = event.__class__
        processors = await self.event_processors.get_processors(event_type)
        for processor in processors:
            processed_event = await processor.process(event)
            if processed_event is not None:
                await self.event_loop.append(processed_event)
