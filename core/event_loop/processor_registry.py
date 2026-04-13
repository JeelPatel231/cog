from typing import Sequence

from core.event_loop.single_event_processor import SingleEventProcessor
from core.logger import logger

class EventProcessorRegistry:
    def __init__(self, initial_processors: set[SingleEventProcessor] = set()):
        logger.debug(f"Initializing EventProcessorRegistry with processors: {[type(p).__name__ for p in initial_processors]}")
        self.__processors: set[SingleEventProcessor] = initial_processors

    async def register_processor(self, processor: SingleEventProcessor):
        logger.debug(f"Registering processor: {type(processor).__name__}")
        self.__processors.add(processor)

    @property
    def processors(self) -> Sequence[SingleEventProcessor]:
        return list(self.__processors)