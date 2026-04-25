import asyncio

from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from core.event_processors.message import (
    MessageEventProcessor,
    UserReplyEventProcessor,
)
from core.event_processors.subagent import (
    SubAgentThinkingEventProcessor,
)
from core.history_transformer import PassthroughHistoryTransformer
from core.event_loop.processor_registry import EventProcessorRegistry
from core.tools.mcp_adapter import McpToolAdapterImpl
from core.tools.read import ReadTool
from core.tools.run import RunTool
from core.tools.skill import SkillTool
from core.tools.subagent import SubAgent, SubAgentTool


from core.chat import (
    ChatMessage,
    TextMessageContent,
    UserMessage,
)
from core.event_processors.message import MessageEvent
from core.in_memory_event_queue import InMemoryEventQueue
from core.chat.openai_chat import OpenAIChat
from core.event_loop.processor import EventLoopProcessor
from core.tool_provider import InMemoryToolRegistry, ToolProvider

import logging

# logging.basicConfig(level=logging.DEBUG)


async def run_once() -> None:
    queue = InMemoryEventQueue()
    no_op_history_transformer = PassthroughHistoryTransformer()

    tool_registry = InMemoryToolRegistry()
    tool_provider = ToolProvider(tool_registry)
    # agent = OpenRouterChat(tool_provider=tool_provider, model="gpt-4o-mini")
    agent = OpenAIChat(api_key='mock', base_url='http://localhost:8080/v1', tool_provider=tool_provider)

    subagent = SubAgent(agent, tool_provider, no_op_history_transformer)

    await tool_registry.register_tool(
        SubAgentTool(subagent, queue),
        SkillTool,
        ReadTool,
        RunTool,
    )

    input_processors = {
        MessageEventProcessor(
            agent=agent,
            tool_provider=tool_provider,
            history_transformer=no_op_history_transformer,
        ),
    }

    output_processors = {
        UserReplyEventProcessor(),
        SubAgentThinkingEventProcessor(),
    }

    event_processor_registry = EventProcessorRegistry(
        initial_processors=input_processors | output_processors
    )

    processor = EventLoopProcessor(
        event_queue=queue, event_processor_registry=event_processor_registry
    )

    instruction = "I want you to start a subagent to calculate 134168734 times 381246. No hallucinations, only precise answers."

    conversation: list[ChatMessage] = [
        UserMessage(
            role="user",
            content=TextMessageContent(text=instruction),
        )
    ]

    event = MessageEvent(data=conversation)

    await queue.push(event)

    await processor.start()


if __name__ == "__main__":
    asyncio.run(run_once())
