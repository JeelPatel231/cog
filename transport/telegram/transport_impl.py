import asyncio
from collections.abc import AsyncGenerator
from core.logger import logger
from core.chat import AssistantMessage, UserMessage
from core.event_processors.message import IntermediateResponse

class TelegramTransport:
  async def get_events(self) -> AsyncGenerator[UserMessage]:
    # connect to the tg server.
    # grab the inbox and yield out all the events after mapping them.
    raise NotImplementedError()
    while True:
      yield

  async def handle_thinking_output(self, intermediate_response: IntermediateResponse):
      logger.info("[Intermediate] %s", intermediate_response)

  async def handle_final_output(self, output: AssistantMessage):
      logger.info("[FINAL] %s", output)
  