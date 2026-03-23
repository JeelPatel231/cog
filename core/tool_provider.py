import json
from inspect import signature
from typing import Any, Protocol

from pydantic import BaseModel

from core.tools import Tool, ToolResult


class ToolDefinitionExtractor:
	@staticmethod
	def input_model(tool: Tool) -> type[BaseModel]:
		func_params = signature(tool.callback).parameters
		if len(func_params) != 1:
			raise ValueError("Tool callback must have exactly one parameter")

		input_param = next(iter(func_params.values()))
		input_class = input_param.annotation
		if not isinstance(input_class, type) or not issubclass(input_class, BaseModel):
			raise ValueError("Tool callback parameter must be a Pydantic model")

		return input_class

	@staticmethod
	def input_schema[TIn: BaseModel](tool: Tool[TIn]) -> dict[str, Any]:
		return ToolDefinitionExtractor.input_model(tool).model_json_schema()


class ToolRegistry(Protocol):
	async def register_tool(self, tool: Tool) -> None: ...

	async def get_tool(self, tool_name: str) -> Tool: ...

	async def list_tools(self) -> list[Tool]: ...


class InMemoryToolRegistry:
	def __init__(self, initial_tools: list[Tool] = []):
		self._tools: dict[str, Tool] = {tool.name: tool for tool in initial_tools}

	async def register_tool(self, tool: Tool) -> None:
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
		self._tool_registry = tool_registry


  # this should be factored out as this class is only for providing tool definitions, not executing them. 
	# The execution should be handled by a separate class that takes in the tool registry as a dependency.
	async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
		tool = await self._tool_registry.get_tool(tool_name)
		input_model = ToolDefinitionExtractor.input_model(tool)
		parsed_input = input_model.model_validate(arguments)
		return tool.callback(parsed_input)

	async def get_tool_definitions(self) -> list[dict[str, Any]]:
		all_tools = await self._tool_registry.list_tools()
		return [
			{
				"name": tool.name,
				"description": tool.description,
				"input_schema": ToolDefinitionExtractor.input_schema(tool),
			}
			for tool in all_tools
		]

	async def get_system_prompt(self) -> str:
		definitions = await self.get_tool_definitions()
		return (
			"You can use the following tools. "
			"When needed, call them with valid JSON arguments that match the input schema.\n\n"
			f"{json.dumps(definitions, indent=2)}"
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