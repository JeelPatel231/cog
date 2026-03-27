import asyncio
from asyncio import Future
from typing import Any

from pydantic import BaseModel
from functools import partial

from core.chat import TextMessageContent, UserMessage
from core.event_loop.event_queue import EventQueue
from core.event_processors.message import MessageEvent
from core.event_processors.subagent import SubAgentEvent

from . import Tool, ToolResult


class SubAgentInput(BaseModel):
    instruction: str


async def call_subagent(input: SubAgentInput, queue: EventQueue[Any]) -> ToolResult:
    fut = Future()
    await queue.push(
        SubAgentEvent(
            data=[
                UserMessage(
                    role="user", content=TextMessageContent(text=input.instruction)
                )
            ],
            handle=fut,
        )
    )
    return await fut


def SubAgentTool(queue: EventQueue[Any]) -> Tool[SubAgentInput]:
    return Tool[SubAgentInput](
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=partial(call_subagent, queue=queue),
    )
