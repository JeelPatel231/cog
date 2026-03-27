from typing import AsyncIterator, Sequence

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
    data: Sequence[ChatMessage]

@dataclass
class ReplyToUser(OutputEvent):
    data: AssistantMessage 

@dataclass
class DeferredToolCall:
    call_id: str
    name: str
    task_handle: Task


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
    
    async def process(self, event: MessageEvent) -> AsyncIterator[Event]:
        print(f"Processing message event: {event.data}")
        conversation = event.data
        assert conversation, "Conversation cannot be empty"

        last_message = conversation[-1]

        assert not isinstance(
            last_message, AssistantMessage 
        ), "Last message must not be an AssistantMessage"

        model_history = self.history_transformer.transform(conversation)
        response = await self.agent.send_message(model_history)

        tool_calls = response.tool_calls
        
        if not tool_calls:
            # nothing to process. Just return the assistant message as is.
            yield ReplyToUser(data=response)
            return

        tool_result_futures = await asyncio.gather(
            *[
                self.tool_provider.call_tool(tool.name, tool.arguments)
                for tool in tool_calls
            ],
        )

        tool_results = [
            ToolResponseMessage(
                role="tool",
                id=tool.id,
                name=tool.name,
                content=TextMessageContent(text=result.output),
            )
            for tool, result in zip(tool_calls, tool_result_futures)
        ]

        # add the conversation back to loop for next iteration.
        yield MessageEvent(data=[*conversation, response, *tool_results])
