from pathlib import Path
import os

from core.tools import Tool, ToolResult
from core.tools.utils.pydantic_adapter import PydanticToolArgs


def is_valid_absolute_path(path_str: str) -> bool:
    path = Path(path_str)
    if not path.is_absolute():
        return False
    valid_roots = [Path(root) for root in os.environ.get("SKILLS_PATH", "").split(":")]
    return any(path.is_relative_to(root) for root in valid_roots)


class ReadParams(PydanticToolArgs):
    absolute_path: str


async def read(read: ReadParams) -> ToolResult:
    if not is_valid_absolute_path(read.absolute_path):
        raise ValueError(
            "The path is not absolute, or the agent is not authorized to read it"
        )
    with open(read.absolute_path, "r") as f:
        return ToolResult(output=f.read())


ReadTool = Tool(
    name="read",
    description="Reads the contents of a file at a given absolute path.",
    callback=read,
)
