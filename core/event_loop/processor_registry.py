from collections import defaultdict
from typing import Mapping
from core.event_loop.single_event_processor import SingleEventProcessor
from . import Event


from typing import Type

class EventProcessorRegistry:
    def __init__(self, initial_processors: Mapping[Type[Event], set[SingleEventProcessor]] = dict()):
        self.processors: dict[Type[Event], set[SingleEventProcessor]] = defaultdict(set, initial_processors)

    async def register_processor(
        self, event_class: Type[Event], processor: SingleEventProcessor
    ):
        self.processors[event_class].add(processor)

    async def get_processors(self, event_class: Type[Event]) -> set[SingleEventProcessor]:
        return self.processors[event_class]

