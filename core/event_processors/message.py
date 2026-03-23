from typing import Literal, Sequence

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
from core.event_loop import Event
from core.tool_provider import ToolProvider
from dataclasses import dataclass
import asyncio
from asyncio import Task


@dataclass
class MessageEvent(Event):
    data: Sequence[ChatMessage | DeferredToolCallBatch]


@dataclass
class DeferredToolCall:
    call_id: str
    name: str
    task_handle: Task


@dataclass
class DeferredToolCallBatch:
    handles: list[DeferredToolCall]


class MessageEventProcessor(SingleEventProcessor[MessageEvent, MessageEvent]):
    def __init__(
        self,
        agent: ChatProtocol,
        tool_provider: ToolProvider,
        history_transformer: HistoryTransformer,
    ) -> None:
        self.agent = agent
        self.tool_registry = tool_provider
        self.history_transformer = history_transformer

    async def process(self, event: MessageEvent) -> MessageEvent | None:
        print(f"Processing message event: {event.data}")
        conversation = event.data
        assert conversation, "Conversation cannot be empty"

        last_message = conversation[-1]

        # if last message is a user message, pass the conversation to the agent and get a response
        if isinstance(last_message, UserMessage) or (
            isinstance(last_message, ToolResponseMessage)
        ):
            model_history = self.history_transformer.transform(conversation)
            response = await self.agent.send_message(model_history)
            return MessageEvent(data=[*conversation, response])

        if isinstance(last_message, AssistantMessage) and last_message.tool_calls:
            # if the last message is an assistant message with a tool call, we can process the tool call and get the output

            tool_calls = []
            for tool in last_message.tool_calls:
                tool_output = asyncio.create_task(
                    self.tool_registry.call_tool(tool.name, tool.arguments)
                )

                tool_calls.append(
                    DeferredToolCall(
                        call_id=tool.id,
                        name=tool.name,
                        task_handle=tool_output,
                    )
                )

            return MessageEvent(data=[*conversation, DeferredToolCallBatch(handles=tool_calls)])


        if isinstance(last_message, DeferredToolCallBatch):
            all_done = all(deferred_call.task_handle.done() for deferred_call in last_message.handles)
            if not all_done:
                await asyncio.sleep(1)  # wait a bit before checking again, to avoid busy waiting
                # if not all tool calls are done, we can skip processing for now and check back later when the event is re-processed
                return MessageEvent(data=conversation)
            
            
            tool_results: list[ToolResponseMessage] = []
            for deferred_call in last_message.handles:
                try:
                    result = deferred_call.task_handle.result()
                    tool_results.append(ToolResponseMessage(
                        role="tool",
                        name=deferred_call.name,
                        id=deferred_call.call_id,
                        content=TextMessageContent(text=str(result.output)),
                    ))
                except Exception as e:
                    tool_results.append(ToolResponseMessage(
                        role="tool",
                        name=deferred_call.name,
                        id=deferred_call.call_id,
                        content=TextMessageContent(text=f"Error executing tool: {e}"),
                    ))

            return MessageEvent(data=[*conversation[:-1], *tool_results])

        return None
