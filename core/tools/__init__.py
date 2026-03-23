from typing import Callable

from pydantic import BaseModel


class ToolResult(BaseModel):
    output: str


class Tool[TIn: BaseModel](BaseModel):
    name: str
    description: str
    callback: Callable[[TIn], ToolResult]


from .math import AdditionTool

__all__ = ["Tool", "ToolResult", "AdditionTool"]