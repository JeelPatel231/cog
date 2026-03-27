from typing import AsyncIterator, Protocol
from . import Event, InputEvent

class SingleEventProcessor[TIn : InputEvent, TOut : Event](Protocol):
    """
    The processor can output any kind of event, both input and output.

    The output events will be put to the output event queue for further processing by output processors.
    The input events will be put back to the event loop for processing.
    """
    def process(self, event: TIn) -> AsyncIterator[TOut]: ...
