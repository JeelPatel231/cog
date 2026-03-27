from typing import Protocol
from . import Event

class EventQueue[T: Event](Protocol):
    """
    this is a simple queue. it can be implemented with an in-memory list/asyncio queue, or with a more robust solution like Redis or RabbitMQ. 
    the implementation details are not important, as long as it supports appending and popping events.
    """
    async def push(self, item: T) -> None: ...
    async def pop(self) -> T: ...