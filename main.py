import asyncio

from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

from core.chat import (
    ChatMessage,
    ImageMessageContent,
    TextMessageContent,
    ToolCallMessageContent,
    UserMessage,
)
from core.event_loop import Event
from core.in_memory_event_loop import InMemoryEventLoop
from core.openrouter_chat import OpenRouterChat
from core.processor import EventLoopProcessor
from core.tool_provider import InMemoryToolRegistry, ToolProvider
from core.tools import AdditionTool


async def run_once() -> None:
    queue = InMemoryEventLoop()
    tool_registry = InMemoryToolRegistry()
    await tool_registry.register_tool(AdditionTool)
    tool_provider = ToolProvider(tool_registry)

    agent = OpenRouterChat(tool_provider=tool_provider)
    processor = EventLoopProcessor(event_loop=queue, agent=agent, tool_provider=tool_provider)

    conversation: list[ChatMessage] = [
        UserMessage(role="user", content=TextMessageContent(text="whats 23897 + 98234? i want precise answer, not an approximation")),
    ]
    event = Event(type="chat", data=conversation)

    await queue.append(event)
    next_event = await queue.pop()
    await processor.process(next_event)

    response = next_event.data[-1]
    content = response.content

    if isinstance(content, TextMessageContent):
        print(content.text)
    elif isinstance(content, ToolCallMessageContent):
        print(content.output)
    elif isinstance(content, ImageMessageContent):
        print("[assistant returned an image payload]")
    else:
        print("[assistant returned unknown content]")


def main() -> None:
    try:
        asyncio.run(run_once())
    except ValueError as error:
        if "OPENROUTER_API_KEY" in str(error):
            print("OPENROUTER_API_KEY is required to run this example.")
            return
        raise
    except RuntimeError as error:
        print(f"Run failed: {error}")
        return


if __name__ == "__main__":
    main()
