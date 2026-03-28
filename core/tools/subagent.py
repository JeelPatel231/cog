import asyncio
from asyncio import Future
from typing import Any

from pydantic import BaseModel

from core.chat import TextMessageContent, UserMessage
from core.event_loop.event_queue import EventQueue
from core.event_processors.message import MessageEvent
from core.event_processors.subagent import SubAgentEvent
from core.tools.utils.pydantic_adapter import PydanticToolArgs

from . import Tool, ToolResult


class SubAgentInput(PydanticToolArgs):
    instruction: str


def call_subagent(queue: EventQueue[Any]):
    async def inner(input: SubAgentInput) -> ToolResult:
        fut = Future[str]()
        print(f"Calling sub-agent with instruction: {input.instruction}")
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
        response = await fut
        return ToolResult(output=response)
    
    return inner

def SubAgentTool(queue: EventQueue[Any]) -> Tool[SubAgentInput]:
    return Tool[SubAgentInput](
        name="subagent",
        description="Delegate tasks to a sub-agent.",
        callback=call_subagent(queue), 
    )
