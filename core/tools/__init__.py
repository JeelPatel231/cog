from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from pydantic import BaseModel


# TODO: tools should be able to return much more than just strings. like images.
class ToolResult(BaseModel):
    output: str

@dataclass(frozen=True)
class Tool[T: dict]:
    name: str
    description: str
    callback: Callable[[Optional[T]], Awaitable[ToolResult]]
    args_json_schema: Optional[dict[str, Any]]

