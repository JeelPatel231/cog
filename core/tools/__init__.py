from typing import Callable
from typing import Any, Protocol, Callable

from pydantic import BaseModel
from inspect import signature

from core.tools import Tool, ToolResult


from pydantic import BaseModel


class ToolResult:
    def __init__(self, output: str):
        self.output = output


class Tool[TIn: BaseModel](BaseModel):
    name: str
    description: str
    callback: Callable[[TIn], ToolResult]


class ToolDefinitionExtractor:
    @staticmethod
    def input_schema[TIn: BaseModel](
        tool: Callable[[TIn], ToolResult],
    ) -> dict[str, Any]:
        func_params = signature(tool).parameters
        assert len(func_params) == 1, "Tool callback must have exactly one parameter"
        input_param = next(iter(func_params.values()))
        input_class = input_param.annotation
        assert issubclass(
            input_class, BaseModel
        ), "Tool callback parameter must be a Pydantic model"

        return input_class.model_json_schema()


class ToolRegistry(Protocol):
    async def register_tool(self, tool: Tool): ...

    async def get_tool(self, tool_name: str) -> Tool: ...

    async def list_tools(self) -> list[Tool]: ...

class InMemoryToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    async def register_tool(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    async def get_tool(self, tool_name: str) -> Tool:
        return self.tools[tool_name]

    async def list_tools(self) -> list[Tool]:
        return list(self.tools.values())

class ToolProvider:
    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    async def call_tool(self, tool_name: str, *args, **kwargs) -> ToolResult:
        tool = await self.tool_registry.get_tool(tool_name)
        return tool.callback(*args, **kwargs)

    async def get_tool_definitions(self) -> list[dict[str, Any]]:
        all_tools = await self.tool_registry.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": ToolDefinitionExtractor.input_schema(tool.callback),
            }
            for tool in all_tools
        ]


tool_registry = InMemoryToolRegistry()
tool_provider = ToolProvider(tool_registry)