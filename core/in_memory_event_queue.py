import asyncio

from core.event_loop.event_queue import EventQueue
from core.event_loop import Event


class InMemoryEventQueue(EventQueue):
    def __init__(self):
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def append(self, item: Event) -> None:
        await self._queue.put(item)

    async def pop(self) -> Event:
        return await self._queue.get()
