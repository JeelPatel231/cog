from typing import AsyncIterator, Optional, Protocol, TypeGuard
from . import Event


class SingleEventProcessor[TIn: Event, TOut: Event](Protocol):
    """
    The processor can output any kind of event, both input and output.

    The output events will be put to the output event queue for further processing by output processors.
    The input events will be put back to the event loop for processing.
    """
    async def can_process(self, event: Event) -> TypeGuard[TIn]: ...

    def process(self, event: TIn) -> AsyncIterator[Optional[TOut]]: ...
