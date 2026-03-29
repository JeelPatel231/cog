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
from dataclasses import dataclass
import asyncio
from asyncio import Task


@dataclass
class MessageEvent(InputEvent):
    data: Sequence[ChatMessage | DeferredToolCallBatch]


@dataclass
class ReplyToUser(OutputEvent):
    data: str


@dataclass
class DeferredToolCall:
    call_id: str
    name: str
    task_handle: Task


@dataclass
class DeferredToolCallBatch:
    calls: Sequence[DeferredToolCall]


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
            print(f"User/Tool message received: {last_message}")

            model_history = self.history_transformer.transform(conversation)
            response = await self.agent.send_message(model_history)

            tool_calls = response.tool_calls
            
            assert isinstance(response.content, TextMessageContent), "Expected TextMessageContent in response"

            if response.content.text:
                yield ReplyToUser(data=response.content.text)

            if not tool_calls:
                # nothing to process.
                return

            tool_result_futures = [
                asyncio.create_task(
                    self.tool_provider.call_tool(tool.name, tool.arguments)
                )
                for tool in tool_calls
            ]

            if tool_result_futures:
                deferred_calls = DeferredToolCallBatch(
                    calls=[
                        DeferredToolCall(
                            call_id=tool_call.id,
                            name=tool_call.name,
                            task_handle=task,
                        )
                        for tool_call, task in zip(tool_calls, tool_result_futures)
                    ]
                )
                yield MessageEvent(data=[*conversation, response, deferred_calls])
                return


        if isinstance(last_message, DeferredToolCallBatch):
            await asyncio.sleep(1)
            tool_calls = last_message.calls
            all_done = any(call.task_handle.done() for call in tool_calls)

            if not all_done:
                # if none of the tool calls are done, we can skip processing for now.
                yield event
                return

            outputs = [call.task_handle.result() for call in tool_calls]

            tool_results = [
                ToolResponseMessage(
                    role="tool",
                    id=tool.call_id,
                    name=tool.name,
                    content=TextMessageContent(text=result.output),
                )
                for tool, result in zip(tool_calls, outputs)
            ]

            # add the conversation back to loop for next iteration.
            yield MessageEvent(data=[*conversation[:-1], *tool_results])
            return

        raise ValueError(f"Unsupported message type: {type(last_message)}")

    
class UserReplyEventProcessor(SingleEventProcessor[ReplyToUser, Event]):
    async def can_process(self, event: Event) -> TypeGuard[ReplyToUser]:
        return isinstance(event, ReplyToUser)

    async def process(self, event: ReplyToUser) -> AsyncIterator[None]:
        if False: yield
        print(f"Assistant: {event.data}")
        return