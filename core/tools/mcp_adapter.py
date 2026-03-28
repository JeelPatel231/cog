from typing import Protocol, Self

from core.tools import Tool, ToolResult
from fastmcp import Client
from mcp.types import Tool as McpTool
from jsonschema import validate

class McpToolAdapter(Protocol):
    async def get_tools(self) -> list[Tool]: ... 


class McpToolAdapterImpl(McpToolAdapter):
    def __init__(self, mcp_url: str):
        self.url = mcp_url
        self.client = Client(mcp_url)

    def __create_tool(self, mcp_tool: McpTool):

        class JsonSchemaToolArgs:
            def __init__(self, arguments: dict):
                self.validated_args = arguments

            @classmethod
            def tool_json_schema(cls) -> dict:
                return mcp_tool.inputSchema
                
            @classmethod
            def tool_validate(cls, arguments: dict) -> Self:
                validate(instance=arguments, schema=cls.tool_json_schema())
                return cls(arguments)

        async def callback(args: JsonSchemaToolArgs) -> ToolResult:
            arguments = args.validated_args
            print(f"Calling MCP tool {mcp_tool.name} with arguments: {arguments}")
            raise NotImplementedError("Calling tools is not implemented yet.")

        return Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            callback=callback,
        )

    async def get_tools(self) -> list[Tool]:
      async with self.client as client:
          tools = await client.list_tools()
          # resources = await client.list_resources()
          # prompts = await client.list_prompts()

          return [self.__create_tool(tool) for tool in tools]