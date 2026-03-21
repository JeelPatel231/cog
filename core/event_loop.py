from typing import Protocol
from pydantic import BaseModel

from core.chat import ChatMessage

class Event(BaseModel):
    type: str
    data: list[ChatMessage]

class EventLoop(Protocol):
    """
    this is a simple queue. it can be implemented with an in-memory list/asyncio queue, or with a more robust solution like Redis or RabbitMQ. 
    the implementation details are not important, as long as it supports appending and popping events.
    """
    async def append(self, item: Event) -> None: ...
    async def pop(self) -> Event: ...