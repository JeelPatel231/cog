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
from core.chat.openai_chat import OpenAIChat
from core.event_loop.processor import TransportEventProcessor
from core.tool_provider import InMemoryToolRegistry, ToolProvider
from transport.telegram.transport_impl import TelegramTransport

import logging

# logging.basicConfig(level=logging.DEBUG)


async def run_once() -> None:
    no_op_history_transformer = PassthroughHistoryTransformer()

    tool_registry = InMemoryToolRegistry()
    tool_provider = ToolProvider(tool_registry)
    # agent = OpenRouterChat(tool_provider=tool_provider, model="gpt-4o-mini")
    agent = OpenAIChat(
        api_key="mock", base_url="http://localhost:8080/v1", tool_provider=tool_provider
    )

    subagent = SubAgent(agent, tool_provider, no_op_history_transformer)

    await tool_registry.register_tool(
        SubAgentTool(subagent),
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

    # these will be gone. maybe some kind of mapped classl will be used for external usage.
    # for internal outputs, we still use this output processors.
    output_processors = {
        UserReplyEventProcessor(),
        SubAgentThinkingEventProcessor(),
    }

    event_processor_registry = EventProcessorRegistry(
        initial_processors=input_processors #| output_processors
    )

    telegram_transport = TelegramTransport()
    processor = TransportEventProcessor(
        transport=telegram_transport, event_processor_registry=event_processor_registry
    )

    instruction = "calculate 134168734 times 381246 using math skill. No hallucinations, only precise answers."

    conversation: list[ChatMessage] = [
        UserMessage(
            role="user",
            content=TextMessageContent(text=instruction),
        )
    ]

    event = MessageEvent(data=conversation)

    await processor.fire_event(event)


if __name__ == "__main__":
    asyncio.run(run_once())
