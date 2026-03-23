import asyncio

from pydantic import BaseModel

from . import Tool, ToolResult

class AdditionInput(BaseModel):
    a: int
    b: int

async def add(input: AdditionInput) -> ToolResult:
    await asyncio.sleep(10)  # simulate some processing time
    return ToolResult(output=str(input.a + input.b))

AdditionTool = Tool[AdditionInput](
    name="addition",
    description="Add two integers together.",
    callback=add,
)