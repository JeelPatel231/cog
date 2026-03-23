from typing import Protocol, TypeVar
from . import Event

class SingleEventProcessor[TIn : Event, TOut : Event](Protocol):
    async def process(self, event: TIn) -> TOut|None: ...
