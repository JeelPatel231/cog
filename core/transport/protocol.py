from collections.abc import AsyncGenerator
from typing import Protocol, runtime_checkable
from core.event_processors.message import IntermediateResponse
from core.chat import AssistantMessage, UserMessage

@runtime_checkable
class Transport(Protocol):
  def get_events(self) -> AsyncGenerator[UserMessage]: ... 

  async def handle_thinking_output(self, intermediate_response: IntermediateResponse): ...

  async def handle_final_output(self, output: AssistantMessage): ...