import json

from core.chat import ChatProtocol
from core.event_loop import EventLoop, Event
from core.processor_registry import EventProcessorRegistry
from core.tool_provider import ToolProvider


class EventLoopProcessor:
    def __init__(
        self,
        event_loop: EventLoop,
        agent: ChatProtocol,
        tool_provider: ToolProvider,
        event_processor_registry: EventProcessorRegistry,
    ):
        self.event_loop = event_loop
        self.agent = agent
        self.tool_provider = tool_provider
        self.event_processors = event_processor_registry

    async def process(self, event: Event):
        """
        take an event from the loop. the event will be a serialized conversation containing user and assistant messages.
        the set of messages will be passed to the agent, which will return a response. the response will be appended to the conversation.
        the new conversation will be serialized and put back on the event loop for the next processor to handle.

        the updated conversation should only be put back in the event loop if there are tool calls being made.
        """
        event_type = event.type
        processors = await self.event_processors.get_processors(event_type)
        for processor in processors:
            processed_event = await processor.process(event)
            if processed_event is not None:
                await self.event_loop.append(processed_event)

    async def _execute_tool_calls(self, raw_tool_calls: str) -> list[str]:
        tool_calls = json.loads(raw_tool_calls)
        results: list[str] = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or ""
            raw_arguments = tool_call.get("arguments") or "{}"

            if isinstance(raw_arguments, str):
                arguments = json.loads(raw_arguments)
            elif isinstance(raw_arguments, dict):
                arguments = raw_arguments
            else:
                arguments = {}

            try:
                result = await self.tool_provider.call_tool(tool_name, arguments)
                results.append(f"Tool '{tool_name}' result: {result.output}")
            except Exception as error:
                results.append(f"Tool '{tool_name}' error: {error}")

        return results
