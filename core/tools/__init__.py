from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol, Self, runtime_checkable

from pydantic import BaseModel


# TODO: tools should be able to return much more than just strings. like images.
class ToolResult(BaseModel):
    output: str

@runtime_checkable
class ToolArgs(Protocol):
    def tool_json_schema(self) -> dict[str, Any]: ...
    def tool_validate(self, arguments: dict[str, Any]) -> Self: ...

@dataclass(frozen=True)
class Tool[T: ToolArgs]:
    name: str
    description: str
    callback: Callable[[T], Awaitable[ToolResult]]

