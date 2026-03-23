import asyncio

from dotenv import load_dotenv

from core.event_processors.message import MessageEventProcessor
from core.history_transformer import PassthroughHistoryTransformer
from core.processor_registry import EventProcessorRegistry

load_dotenv()  # Load environment variables from .env file

from core.chat import (
    ChatMessage,
    ImageMessageContent,
    TextMessageContent,
    UserMessage,
)
from core.event_loop import MessageEvent
from core.in_memory_event_loop import InMemoryEventLoop
from core.openrouter_chat import OpenRouterChat
from core.processor import EventLoopProcessor
from core.tool_provider import InMemoryToolRegistry, ToolProvider
from core.tools import AdditionTool


async def run_once() -> None:
    queue = InMemoryEventLoop()
    tool_registry = InMemoryToolRegistry(initial_tools=[AdditionTool])
    tool_provider = ToolProvider(tool_registry)

    agent = OpenRouterChat(tool_provider=tool_provider, model="gpt-4o-mini")
    event_processor_registry = EventProcessorRegistry(
        initial_processors={
            MessageEvent: {
                MessageEventProcessor(
                    agent=agent,
                    tool_provider=tool_provider,
                    history_transformer=PassthroughHistoryTransformer(),
                )
            }
        }
    )

    processor = EventLoopProcessor(
        event_loop=queue, event_processor_registry=event_processor_registry
    )

    conversation: list[ChatMessage] = [
        UserMessage(
            role="user",
            content=TextMessageContent(
                text="whats 23897 + 983412 i want precise answer, not an approximation. make sure to use tools."
            ),
        )
    ]
    event = MessageEvent(type="chat", data=conversation)

    await queue.append(event)
    while next_event := await queue.pop():
        print(f"Next event: {next_event}")
        await processor.process(next_event)

    response = next_event.data[-1]
    content = response.content

    print(content)


if __name__ == "__main__":
    asyncio.run(run_once())
