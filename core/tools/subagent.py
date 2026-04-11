import asyncio
from asyncio import Future
from typing import Any, AsyncIterator, Sequence

from pydantic import BaseModel, ConfigDict

from core.chat import (
    ChatMessage,
    ChatProtocol,
    TextMessageContent,
    ToolResponseMessage,
    UserMessage,
)
from core.event_loop import Event
from core.event_loop.event_queue import EventQueue
from core.event_processors.message import MessageEvent
from core.event_processors.subagent import SubAgentThinkingOutput
from core.history_transformer import HistoryTransformer
from core.tool_provider import ToolProvider
from core.tools.utils.pydantic_adapter import PydanticToolArgs

from . import Tool, ToolResult


class SubAgentResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: bool
    output: str


class SubAgentInput(PydanticToolArgs):
    instruction: str


class SubAgent:
    def __init__(
        self,
        agent: ChatProtocol,
        tool_provider: ToolProvider,
        history_transformer: HistoryTransformer,
    ) -> None:
        self.agent = agent
        self.tool_provider = tool_provider
        self.history_transformer = history_transformer

    async def do_task(
        self, conversation: Sequence[ChatMessage]
    ) -> AsyncIterator[SubAgentThinkingOutput]:
        while True:
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
                self.tool_provider.call_tool(tool.name, tool.arguments)
                for tool in tool_calls
            ]
            outputs = await asyncio.gather(*tool_result_futures)
            tool_results = [
                ToolResponseMessage(
                    role="tool",
                    id=tool.id,
                    name=tool.name,
                    content=TextMessageContent(text=result.output),
                )
                for tool, result in zip(tool_calls, outputs)
            ]

            for tool_result in tool_results:
                print(
                    f"Tool '{tool_result.name}' returned: {tool_result.content.text}"
                )

            conversation = [*conversation, response, *tool_results]

def call_subagent(subagent: SubAgent, event_queue: EventQueue):
    async def inner(input: SubAgentInput) -> ToolResult:
        print(f"Calling sub-agent with instruction: {input.instruction}")
        conversation = [
            UserMessage(role="user", content=TextMessageContent(text=input.instruction))
        ]
        last_event = None
        async for event in subagent.do_task(conversation):
            last_event = event
            await event_queue.push(event)
        
        assert last_event is not None

        return ToolResult(output=last_event.data)

    return inner


def SubAgentTool(subagent: SubAgent, event_queue: EventQueue) -> Tool[SubAgentInput]:
    return Tool[SubAgentInput](
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=call_subagent(subagent, event_queue),
    )
