from pydantic import BaseModel
from functools import partial

from core.chat import TextMessageContent, UserMessage
from core.event_loop.loop import EventLoop
from core.event_processors.message import MessageEvent

from . import Tool, ToolResult

class SubAgentInput(BaseModel):
    instruction: str

async def call_subagent(event_loop: EventLoop, input: SubAgentInput) -> ToolResult:
    await event_loop.append(
        MessageEvent(data=[UserMessage(role="user", content=TextMessageContent(text=input.instruction))])
    )
    
    raise NotImplementedError("Awaiting sub-agent callback is not implemented")


def SubAgentTool(event_loop: EventLoop) -> Tool[SubAgentInput]:
    return Tool[SubAgentInput](
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=partial(call_subagent, event_loop=event_loop),
    )
