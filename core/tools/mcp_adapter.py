from typing import Any, Optional, Protocol

from core.tools import Tool, ToolResult
from fastmcp import Client
from mcp.types import TextContent, Tool as McpTool

class McpToolAdapter(Protocol):
    async def get_tools(self) -> list[Tool]: ... 


class McpToolAdapterImpl(McpToolAdapter):
    def __init__(self, mcp_url: str):
        self.url = mcp_url
        self.client = Client(mcp_url)

    def __create_tool(self, client: Client, mcp_tool: McpTool):
        async def callback(arguments: Optional[dict[str, Any]]) -> ToolResult:
            async with client:
                result = await client.call_tool(mcp_tool.name, arguments)
            
            text_result = ''.join([x.text for x in result.content if isinstance(x, TextContent)])
            return ToolResult(output=text_result)

        return Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            callback=callback,
            args_json_schema=mcp_tool.inputSchema
        )

    async def get_tools(self) -> list[Tool]:
        async with self.client as client:
            tools = await client.list_tools()
            # resources = await client.list_resources()
            # prompts = await client.list_prompts()

        return [self.__create_tool(self.client, tool) for tool in tools]