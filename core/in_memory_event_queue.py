import asyncio

from core.event_loop.event_queue import EventQueue
from core.event_loop import Event

from core.logger import logger

class InMemoryEventQueue(EventQueue):
    def __init__(self):
        logger.debug("Initializing InMemoryEventQueue")
        self._queue: asyncio.Queue[Event] = asyncio.Queue()

    async def push(self, item: Event) -> None:
        logger.debug("Pushing event to InMemoryEventQueue: %s", item)
        await self._queue.put(item)

    async def pop(self) -> Event:
        event = await self._queue.get()
        logger.debug("Popped event from InMemoryEventQueue: %s", event)
        return event
