from pydantic import BaseModel
from functools import partial

from core.chat import TextMessageContent, UserMessage
from core.event_loop.event_queue import EventQueue
from core.event_processors.message import MessageEvent

from . import Tool, ToolResult

class SubAgentInput(BaseModel):
    instruction: str

async def call_subagent(input: SubAgentInput) -> ToolResult:
    raise NotImplementedError("Awaiting sub-agent callback is not implemented")


def SubAgentTool() -> Tool[SubAgentInput]:
    return Tool[SubAgentInput](
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=call_subagent,
    )
