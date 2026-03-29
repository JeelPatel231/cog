from asyncio import Future, Task

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Sequence, TypeGuard

from pydantic import BaseModel, ConfigDict

from core.event_loop import Event, InputEvent, OutputEvent
from core.event_loop.single_event_processor import SingleEventProcessor

from core.chat import (
    ChatMessage,
    ChatProtocol,
    AssistantMessage,
    TextMessageContent,
    UserMessage,
    ToolResponseMessage,
)
from core.history_transformer import HistoryTransformer
from core.tool_provider import ToolProvider


@dataclass
class SubAgentThinkingOutput(OutputEvent):
    data: str


@dataclass
class DeferredToolCall:
    call_id: str
    name: str
    task_handle: Task


@dataclass
class DeferredToolCallBatch:
    calls: Sequence[DeferredToolCall]


@dataclass
class SubAgentEvent(InputEvent):
    handle: Future[str]
    data: Sequence[ChatMessage | DeferredToolCallBatch]


class SubAgentResponseFormat(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed: bool
    output: str

class SubagentEventProcessor(SingleEventProcessor[SubAgentEvent, Event]):
    def __init__(
        self,
        agent: ChatProtocol,
        tool_provider: ToolProvider,
        history_transformer: HistoryTransformer,
    ) -> None:
        self.agent = agent
        self.tool_provider = tool_provider
        self.history_transformer = history_transformer

    async def can_process(self, event: Event) -> TypeGuard[SubAgentEvent]:
        return isinstance(event, SubAgentEvent)
    
    async def process(self, event: SubAgentEvent) -> AsyncIterator[Event]:
        conversation = event.data
        assert conversation, "Conversation cannot be empty"

        last_message = conversation[-1]

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

            for tool_result in tool_results:
                print(f"Tool '{tool_result.name}' returned: {tool_result.content.text}")

            # add the conversation back to loop for next iteration.
            yield SubAgentEvent(handle=event.handle, data=[*conversation[:-1], *tool_results])
            return



        model_history = self.history_transformer.transform(conversation)
        response = await self.agent.send_message(model_history, response_format=SubAgentResponseFormat)
        
        assert isinstance(response.content, TextMessageContent), f"Expected response content to be TextMessageContent, got {type(response.content)}"
        
        if response.content.text:
            parsed_response = SubAgentResponseFormat.model_validate_json(response.content.text)
            completed = parsed_response.completed

            if parsed_response.output:
                yield SubAgentThinkingOutput(data=parsed_response.output)
            
            if completed:
                event.handle.set_result(parsed_response.output)
                return

        tool_calls = response.tool_calls

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
            yield SubAgentEvent(handle=event.handle, data=[*conversation, response, deferred_calls])
            return


        raise ValueError(f"Unsupported message type: {type(last_message)}")

class SubAgentThinkingEventProcessor(SingleEventProcessor[SubAgentThinkingOutput, Event]):
    async def can_process(self, event: Event) -> TypeGuard[SubAgentThinkingOutput]:
        return isinstance(event, SubAgentThinkingOutput)

    async def process(self, event: SubAgentThinkingOutput) -> AsyncIterator[None]:
        if False: yield

        print(f"SubAgent Output: {event.data}")
        return