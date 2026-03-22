from collections import defaultdict
from typing import Protocol
from .event_loop import Event


class SingleEventProcessor(Protocol):
    async def process(self, event: Event) -> Event: ...


class EventProcessorRegistry:
    def __init__(self):
        self.processors: dict[str, set[SingleEventProcessor]] = defaultdict(set)

    async def register_processor(
        self, event_class: str, processor: SingleEventProcessor
    ):
        self.processors[event_class].add(processor)

    async def get_processors(self, event_class: str) -> set[SingleEventProcessor]:
        return self.processors[event_class]

