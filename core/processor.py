from core.chat import ChatProtocol, ToolCallMessageContent
from core.event_loop import EventLoop, Event


class EventLoopProcessor:
    def __init__(self, event_loop: EventLoop, agent: ChatProtocol):
        self.event_loop = event_loop
        self.agent = agent

    async def process(self, event: Event):
        """
        take an event from the loop. the event will be a serialized conversation containing user and assistant messages.
        the set of messages will be passed to the agent, which will return a response. the response will be appended to the conversation.
        the new conversation will be serialized and put back on the event loop for the next processor to handle.

        the updated conversation should only be put back in the event loop if there are tool calls being made.
        """
        # deserialize the conversation
        conversation = event.data

        # pass the conversation to the agent and get a response
        response = await self.agent.send_message(conversation)

        # append the response to the conversation
        conversation.append(response)

        if not isinstance(response.content, ToolCallMessageContent):
          # no tools to call, stop processing this conversation and don't put it back on the event loop
          return

        # serialize the new conversation and put it back on the event loop
        new_event = Event(type=event.type, data=conversation)
        await self.event_loop.append(new_event)