from typing import Awaitable, Callable

from pydantic import BaseModel


# TODO: tools should be able to return much more than just strings. like images.
class ToolResult(BaseModel):
    output: str


class Tool[TIn: BaseModel](BaseModel):
    name: str
    description: str
    callback: Callable[[TIn], Awaitable[ToolResult]]

