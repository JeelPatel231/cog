import asyncio

from pathlib import Path
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
from core.skills import Skill, SkillRegistry
from core.tools.skill_loader import SkillLoaderTool, SkillRunnerTool
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
    skill_registry = SkillRegistry()

    skill_registry.register(skill=Skill(
        name="math",
        description="A skill for performing mathematical calculations.",
        skill_dir=Path("skills/math"),
    ))

    tool_registry = InMemoryToolRegistry(
        initial_tools=[
            SubAgentTool(queue),
            SkillLoaderTool(skill_registry),
            SkillRunnerTool(skill_registry),
        ]
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

    instruction = "whats 23897 + 983412? delegate the task to a sub-agent and ask it to use the skill loader tool to load math skill for precise calculations."

    conversation: list[ChatMessage] = [
        UserMessage(
            role="user",
            content=TextMessageContent(
                text=instruction
            ),
        )
    ]

    event = MessageEvent(data=conversation)

    await queue.push(event)

    await processor.start()


if __name__ == "__main__":
    asyncio.run(run_once())
