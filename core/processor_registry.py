from collections import defaultdict
from typing import Mapping, Protocol
from .event_loop import Event


class SingleEventProcessor[TIn : Event, TOut : Event](Protocol):
    async def process(self, event: TIn) -> TOut|None: ...


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

