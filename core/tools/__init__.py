from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, Callable
from core.event_loop import OutputEvent


# TODO: tools should be able to return much more than just strings. like images.
@dataclass
class ToolResult(OutputEvent):
    output: str

@dataclass(frozen=True)
class Tool[T: dict]:
    name: str
    description: str
    callback: Callable[[T|None], AsyncGenerator[OutputEvent]]
    args_json_schema: dict[str, Any]|None
