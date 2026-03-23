from typing import Awaitable, Callable

from pydantic import BaseModel


class ToolResult(BaseModel):
    output: str


class Tool[TIn: BaseModel](BaseModel):
    name: str
    description: str
    callback: Callable[[TIn], Awaitable[ToolResult]]

