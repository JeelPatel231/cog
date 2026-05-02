import asyncio
from typing import Any
from collections.abc import AsyncGenerator, AsyncIterator, Sequence

from aiostream.stream import merge, skiplast
from pydantic import BaseModel, ConfigDict

from core.chat import (
    ChatMessage,
    ChatProtocol,
    TextMessageContent,
    ToolResponseMessage,
    UserMessage,
)
from core.chat import SystemMessage, TextMessageContent, UserMessage
from core.event_loop import OutputEvent
from core.event_processors.subagent import SubAgentThinkingOutput
from core.history_transformer import HistoryTransformer
from core.iterator.cache import LastValueIterator
from core.tool_provider import ToolProvider

from . import Tool, ToolResult


class SubAgentResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: bool
    output: str


class SubAgentInput(BaseModel):
    instruction: str

class SubAgent:
    def __init__(
        self,
        agent: ChatProtocol,
        tool_provider: ToolProvider,
        history_transformer: HistoryTransformer,
        max_iterations: int = 10,
    ) -> None:
        self.agent = agent
        self.tool_provider = tool_provider
        self.history_transformer = history_transformer
        self.max_iterations = max_iterations

    async def do_task(
        self, conversation: Sequence[ChatMessage]
    ) -> AsyncIterator[SubAgentThinkingOutput]:
        for _ in range(self.max_iterations):
            model_history = self.history_transformer.transform(conversation)

            response = await self.agent.send_message(
                model_history, response_format=SubAgentResponseFormat
            )

            assert isinstance(
                response.content, TextMessageContent
            ), f"Expected response content to be TextMessageContent, got {type(response.content)}"

            if response.content.text:
                parsed_response = SubAgentResponseFormat.model_validate_json(
                    response.content.text
                )
                completed = parsed_response.completed

                if parsed_response.output:
                    yield SubAgentThinkingOutput(data=parsed_response.output)

                if completed:
                    return

            tool_calls = response.tool_calls

            tool_result_futures = [
                await self.tool_provider.call_tool(tool.name, tool.arguments)
                for tool in tool_calls
            ]

            async with merge(*tool_result_futures).stream() as merged_tool_calls:
                async for event in merged_tool_calls:
                    yield event

            tool_results = [
                ToolResponseMessage(
                    role="tool",
                    id=tool.id,
                    name=tool.name,
                    content=TextMessageContent(text=result.last.output),
                )
                for tool, result in zip(tool_calls, tool_result_futures)
            ]

            for tool_result in tool_results:
                print(f"Tool '{tool_result.name}' returned: {tool_result.content.text}")

            conversation = [*conversation, response, *tool_results]


SUBAGENT_SYSTEM_PROMPT = """
You are a sub-agent. You will receive instructions from the main agent, and you should execute them and provide a final output when done.
You can not communicate with the main agent or the user, you can only receive instructions and provide a final output.

You should reason through all the steps needed to complete the task.
If the task needs extra information, or fails, or is not possible to complete, you provide all the details in the final output so the main agent can decide what to do next.
""".strip()

def call_subagent(subagent: SubAgent):
    async def inner(args: dict[str, Any] | None) -> AsyncGenerator[OutputEvent]:
        input = SubAgentInput.model_validate(args)
        print(f"Calling sub-agent with instruction: {input.instruction}")
        conversation = [
                    SystemMessage(
                        role="system",
                        content=TextMessageContent(
                            text=SUBAGENT_SYSTEM_PROMPT,
                        )
                    ),
            UserMessage(role="user", content=TextMessageContent(text=input.instruction))
        ]

        last_value = LastValueIterator(subagent.do_task(conversation))
        async with skiplast(last_value, n = 1).stream() as streamer:
            async for event in streamer:
                yield event
        
        yield ToolResult(output = last_value.last.data)

    return inner

subagent_tool_description = """
Delegate tasks to a sub-agent by providing instructions.

The spawned agent will be autonomous and it cannot converse with you or user.
It can only receive instructions from you and provide a final output when it completes the task.

This tool is especially useful for delegating complex tasks that require multiple steps or reasoning, 
and you want to offload that work to a separate agent, in order to keep your context window clean and 
focused on high-level instructions and final outputs.
"""

def SubAgentTool(subagent: SubAgent):
    return Tool(
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=call_subagent(subagent),
        args_json_schema=SubAgentInput.model_json_schema()
    )
