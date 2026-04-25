from typing import AsyncIterator, Sequence, TypeGuard

from core.chat import (
    ChatMessage,
    ChatProtocol,
    AssistantMessage,
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
    ) -> None:
        self.agent = agent
        self.tool_provider = tool_provider
        self.history_transformer = history_transformer
    
    async def can_process(self, event: Event) -> TypeGuard[MessageEvent]:
        return isinstance(event, MessageEvent)

    async def process(self, event: MessageEvent) -> AsyncIterator[Event]:
        conversation = event.data
        assert conversation, "Conversation cannot be empty"

        last_message = conversation[-1]

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

            logger.debug(f"All tool call tasks completed")
            # add the conversation back to loop for next iteration.
            yield MessageEvent(data=[*conversation, response, *tool_results])
            return

        raise ValueError(f"Unsupported message type: {type(last_message)}")

    
class UserReplyEventProcessor(SingleEventProcessor[ReplyToUser, Event]):
    async def can_process(self, event: Event) -> TypeGuard[ReplyToUser]:
        return isinstance(event, ReplyToUser)

    async def process(self, event: ReplyToUser) -> AsyncIterator[Event]:
        print(f"Assistant: {event.data}")
        return
        yield