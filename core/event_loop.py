from typing import Protocol, Literal
from pydantic import BaseModel

from core.chat import ChatMessage

class MessageEvent(BaseModel):
    type: Literal["chat"] = "chat"
    data: list[ChatMessage]

Event = MessageEvent

class EventLoop(Protocol):
    """
    this is a simple queue. it can be implemented with an in-memory list/asyncio queue, or with a more robust solution like Redis or RabbitMQ. 
    the implementation details are not important, as long as it supports appending and popping events.
    """
    async def append(self, item: Event) -> None: ...
    async def pop(self) -> Event: ...