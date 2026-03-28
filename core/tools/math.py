from core.tools.utils.pydantic_adapter import PydanticToolArgs

from . import Tool, ToolResult

class AdditionInput(PydanticToolArgs):
    a: int
    b: int

async def add(input: AdditionInput) -> ToolResult:
    return ToolResult(output=str(input.a + input.b))

AdditionTool = Tool(
    name="addition",
    description="Add two integers together.",
    callback=add,
)