import asyncio
from collections.abc import AsyncIterator
from aiostream.stream import skiplast
from core.event_loop.processor_registry import EventProcessorRegistry
from core.iterator.cache import LastValueIterator
from core.transport.protocol import Transport
from . import Event, InputEvent

class TransportEventProcessor:
    def __init__(
        self,
        transport: Transport,
        event_processor_registry: EventProcessorRegistry,
    ):
        self.__transport = transport
        self.__event_processors = event_processor_registry

    async def fire_event(self, event: InputEvent):
        """
        take an event from the loop. the event will be a serialized conversation containing user and assistant messages.
        the set of messages will be passed to the agent, which will return a response. the response will be appended to the conversation.
        the new conversation will be serialized and put back on the event loop for the next processor to handle.

        the updated conversation should only be put back in the event loop if there are tool calls being made.
        """
        for processor in self.__event_processors.processors:
            if not await processor.can_process(event):
                continue

            worker = LastValueIterator(processor.process(event))
            async with skiplast(worker, n = 1).stream() as worker_events:
                async for yielded_event in worker_events:
                    assert isinstance(
                        yielded_event, Event
                    ), f"Processor yielded an item that is not an Event: {yielded_event}"

                    await self.__transport.handle_thinking_output(yielded_event)

            await self.__transport.handle_final_output(worker.last)