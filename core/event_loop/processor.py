from core.event_loop.processor_registry import EventProcessorRegistry
from . import Event, InputEvent, OutputEvent

from .event_queue import EventQueue

class EventLoopProcessor:
    def __init__(
        self,
        input_event_queue: EventQueue[InputEvent],
        output_event_queue: EventQueue[OutputEvent],
        event_processor_registry: EventProcessorRegistry,
    ):
        self.input_event_queue = input_event_queue
        self.output_event_queue = output_event_queue
        self.event_processors = event_processor_registry

    async def start(self):
        """
        take an event from the loop. the event will be a serialized conversation containing user and assistant messages.
        the set of messages will be passed to the agent, which will return a response. the response will be appended to the conversation.
        the new conversation will be serialized and put back on the event loop for the next processor to handle.

        the updated conversation should only be put back in the event loop if there are tool calls being made.
        """
        while True:
            event = await self.input_event_queue.pop()
            event_type = event.__class__
            processors = await self.event_processors.get_processors(event_type)
            
            if not processors:
                print(f"No processors found for event type: {event_type}. Event: {event}")
                continue
        
            for processor in processors:
                coroutine = processor.process(event)
                async for yielded_event in coroutine:
                    assert isinstance(yielded_event, Event), f"Processor yielded an item that is not an Event: {yielded_event}"

                    if isinstance(yielded_event, OutputEvent):
                        # await self.output_event_queue.push(yielded_event)
                        print(f"Output event produced: {yielded_event}")
                        continue
                    
                    if isinstance(yielded_event, InputEvent):
                        await self.input_event_queue.push(yielded_event)
                        continue
                    
                    raise ValueError(f"Processor yielded an event of unsupported type: {type(yielded_event)}")
                