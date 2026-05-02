from typing import TypeGuard, assert_never
from collections.abc import AsyncIterator, Sequence
from aiostream.stream import merge
from core.chat import (
    ChatMessage,
    ChatProtocol,
    AssistantMessage,
    SystemMessage,
    TextMessageContent,
    UserMessage,
    ToolResponseMessage,
)
from core.history_transformer import HistoryTransformer
from core.event_loop.processor_registry import SingleEventProcessor
from core.event_loop import InputEvent, OutputEvent, Event
from core.tool_provider import ToolProvider
from core.logger import logger
from dataclasses import dataclass
import asyncio


@dataclass
class MessageEvent(InputEvent):
    data: Sequence[ChatMessage]


@dataclass
class ReplyToUser(OutputEvent):
    data: str

@dataclass
class IntermediateResponse(OutputEvent):
    data: str

class MessageEventProcessor(SingleEventProcessor[MessageEvent, Event]):
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
    
    async def can_process(self, event: Event) -> TypeGuard[MessageEvent]:
        return isinstance(event, MessageEvent)

    async def process(self, event: MessageEvent) -> AsyncIterator[Event]:
        iterations = 0
        conversation = event.data

        while iterations < self.max_iterations:
            assert conversation, "Conversation cannot be empty"
            last_message = conversation[-1]

            if isinstance(last_message, (AssistantMessage, SystemMessage)):
                logger.error("Assistant/System message must not be sent back to LLM api call. This is potentially an infinite loop.")
                raise RuntimeError('AssistantMessage/SystemMessage is unsupported for processing in agent loop.')

            if isinstance(last_message, (UserMessage, ToolResponseMessage)):
                logger.info(f"User/Tool message received: {last_message}")

                model_history = self.history_transformer.transform(conversation)
                response = await self.agent.send_message(model_history)

                tool_calls = response.tool_calls
                
                assert isinstance(response.content, TextMessageContent), "Expected TextMessageContent in response"

                if response.content.text and tool_calls:
                    yield IntermediateResponse(data=response.content.text)

                if not tool_calls:
                    # termination condition. no more tool calls then the agent is done.
                    assert response.content.text is not None, "Agent didn't call any tool nor has anything to say."
                    yield ReplyToUser(data=response.content.text)
                    return

                tool_result_futures = [
                    await self.tool_provider.call_tool(tool.name, tool.arguments)
                    for tool in tool_calls
                ]

                async with merge(*tool_result_futures).stream() as merged_streams:
                    async for _event in merged_streams:
                        yield _event

                tool_results = [
                    ToolResponseMessage(
                        role="tool",
                        id=tool.id,
                        name=tool.name,
                        content=TextMessageContent(text=result.last.output),
                    )
                    for tool, result in zip(tool_calls, tool_result_futures)
                ]

                logger.debug(f"All tool call tasks completed")

                # add the conversation back to loop for next iteration.
                conversation = [*conversation, response, *tool_results]
                continue

            assert_never(last_message)

    
class UserReplyEventProcessor(SingleEventProcessor[ReplyToUser, Event]):
    async def can_process(self, event: Event) -> TypeGuard[ReplyToUser]:
        return isinstance(event, ReplyToUser)

    async def process(self, event: ReplyToUser) -> AsyncIterator[Event]:
        print(f"Assistant: {event.data}")
        return
        yield