from pydantic import BaseModel

from . import Tool, ToolResult

class AdditionInput(BaseModel):
    a: int
    b: int

def add(input: AdditionInput) -> ToolResult:
    return ToolResult(output=str(input.a + input.b))

AdditionTool = Tool[AdditionInput](
    name="addition",
    description="Add two integers together.",
    callback=add,
)