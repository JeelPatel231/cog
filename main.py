import asyncio

from dotenv import load_dotenv

from core.event_processors.message import (
    MessageEventProcessor,
    ReplyToUser,
    UserReplyEventProcessor,
)
from core.event_processors.subagent import (
    SubAgentEvent,
    SubAgentThinkingEventProcessor,
    SubAgentThinkingOutput,
    SubagentEventProcessor,
)
from core.history_transformer import PassthroughHistoryTransformer
from core.event_loop.processor_registry import EventProcessorRegistry
from core.tools.subagent import SubAgentTool

load_dotenv()  # Load environment variables from .env file

from core.chat import (
    ChatMessage,
    ImageMessageContent,
    TextMessageContent,
    UserMessage,
)
from core.event_processors.message import MessageEvent
from core.in_memory_event_queue import InMemoryEventQueue
from core.openrouter_chat import OpenRouterChat
from core.event_loop.processor import EventLoopProcessor
from core.tool_provider import InMemoryToolRegistry, ToolProvider
from core.tools.math import AdditionTool


async def run_once() -> None:
    queue = InMemoryEventQueue()
    tool_registry = InMemoryToolRegistry(
        initial_tools=[AdditionTool, SubAgentTool(queue)]
    )
    tool_provider = ToolProvider(tool_registry)
    agent = OpenRouterChat(tool_provider=tool_provider, model="gpt-4o-mini")

    input_processors = {
        MessageEvent: {
            MessageEventProcessor(
                agent=agent,
                tool_provider=tool_provider,
                history_transformer=PassthroughHistoryTransformer(),
            )
        },
        SubAgentEvent: {
            SubagentEventProcessor(
                agent=agent,
                tool_provider=tool_provider,
                history_transformer=PassthroughHistoryTransformer(),
            )
        },
    }

    output_processors = {
        ReplyToUser: {
            UserReplyEventProcessor(),
        },
        SubAgentThinkingOutput: {
            SubAgentThinkingEventProcessor(),
        },
    }

    event_processor_registry = EventProcessorRegistry(
        initial_processors=input_processors | output_processors
    )

    processor = EventLoopProcessor(
        event_queue=queue, event_processor_registry=event_processor_registry
    )

    conversation: list[ChatMessage] = [
        UserMessage(
            role="user",
            content=TextMessageContent(
                text="whats 23897 + 983412? i want you to start a subagent to calculate the answer, and report the answer back to me. ask the subagent to use the addition tool to calculate the answer."
            ),
        )
    ]

    event = MessageEvent(data=conversation)

    await queue.push(event)

    await processor.start()


if __name__ == "__main__":
    asyncio.run(run_once())
