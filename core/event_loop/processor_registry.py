from typing import Sequence

from core.event_loop.single_event_processor import SingleEventProcessor

class EventProcessorRegistry:
    def __init__(self, initial_processors: set[SingleEventProcessor] = set()):
        self.__processors: set[SingleEventProcessor] = initial_processors

    async def register_processor(self, processor: SingleEventProcessor):
        self.__processors.add(processor)

    @property
    def processors(self) -> Sequence[SingleEventProcessor]:
        return list(self.__processors)