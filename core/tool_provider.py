import json
from typing import Any, Protocol
from core.tools import Tool, ToolResult


class ToolRegistry(Protocol):
    async def register_tool(self, *tool: Tool) -> None: ...

    async def get_tool(self, tool_name: str) -> Tool: ...

    async def list_tools(self) -> list[Tool]: ...


class InMemoryToolRegistry:
    def __init__(self, initial_tools: list[Tool] = []):
        logger.debug(f"Initializing InMemoryToolRegistry with tools: {[tool.name for tool in initial_tools]}")
        self._tools: dict[str, Tool] = {tool.name: tool for tool in initial_tools}

    async def register_tool(self, *tools: Tool) -> None:
        for tool in tools:
            self._tools[tool.name] = tool

    async def get_tool(self, tool_name: str) -> Tool:
        try:
            return self._tools[tool_name]
        except KeyError as error:
            raise ValueError(f"Tool not found: {tool_name}") from error

    async def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


class ToolProvider:
    def __init__(self, tool_registry: ToolRegistry):
        logger.debug(f"Initializing ToolProvider with registry: {tool_registry}")
        self._tool_registry = tool_registry


    # The execution should be handled by a separate class that takes in the tool registry as a dependency.
    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        logger.debug(f"Calling tool '{tool_name}' with arguments: {arguments}")
        tool = await self._tool_registry.get_tool(tool_name)
        try:
            return await tool.callback(arguments)
        except Exception as error:
            logger.error(f"Error executing tool '{tool_name}': {error}")
            return ToolResult(output=f"Error executing tool '{tool_name}': {error}")

    async def get_tool_definitions(self) -> list[dict[str, Any]]:
        all_tools = await self._tool_registry.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.args_json_schema,
            }
            for tool in all_tools
        ]

    async def get_system_prompt(self) -> str:
        definitions = await self.get_tool_definitions()
        only_info = [
            {"name": definition["name"], "description": definition["description"]}
            for definition in definitions
        ]
        return (
            "You can use the following tools. "
            "When needed, call them with valid JSON arguments that match the input schema.\n\n"
            f"{json.dumps(only_info, indent=2)}"
        )


class ToolCallProcessor(Protocol):
    async def process_tool_call(self, raw_tool_calls: str) -> list[str]: ...


class ToolCallProcessorImpl:
    def __init__(self, tool_provider: ToolProvider):
        self._tool_provider = tool_provider

    async def process_tool_call(self, raw_tool_calls: str) -> list[str]:
        tool_calls = json.loads(raw_tool_calls)
        results: list[str] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or ""
            raw_arguments = tool_call.get("arguments") or "{}"

            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}

            try:
                result = await self._tool_provider.call_tool(tool_name, arguments)
                results.append(f"Tool '{tool_name}' result: {result.output}")
            except Exception as error:
                results.append(f"Tool '{tool_name}' call failed: {error}")

        return results
