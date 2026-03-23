from core.chat import ChatProtocol, AssistantMessage, MessageContent, TextMessageContent, UserMessage,ToolResponseMessage
from core.event_loop import MessageEvent
from core.processor_registry import SingleEventProcessor
from core.tool_provider import ToolProvider

class MessageEventProcessor(SingleEventProcessor[MessageEvent, MessageEvent]):
    def __init__(self, agent: ChatProtocol, tool_provider: ToolProvider) -> None:
        self.agent = agent
        self.tool_registry = tool_provider

    async def process(self, event: MessageEvent) -> MessageEvent|None:
        print(f"Processing message event: {event.data}")
        conversation = event.data
        assert conversation, "Conversation cannot be empty"
        
        last_message = conversation[-1]

        # if last message is a user message, pass the conversation to the agent and get a response
        if isinstance(last_message, UserMessage) or (isinstance(last_message, ToolResponseMessage)):
            # pass the conversation to the agent and get a response
            response = await self.agent.send_message(conversation)
            # append the response to the conversation
            conversation.append(response)
            # return a new MessageEvent with the updated conversation
            return MessageEvent(type="chat", data=conversation)
        
        if isinstance(last_message, AssistantMessage) and last_message.tool_calls:
                # if the last message is an assistant message with a tool call, we can process the tool call and get the output

                for tool in last_message.tool_calls:
                    tool_output = await self.tool_registry.call_tool(tool.name, tool.arguments)
                    # create a new assistant message with the tool output and append it to the conversation
                    # return a new MessageEvent with the updated conversation
                
                    conversation.append(ToolResponseMessage(
                        role="tool", 
                        id=tool.id,
                        name=tool.name,
                        content=TextMessageContent(
                            text=tool_output.model_dump_json()
                        )
                    )
                )

                return MessageEvent(type="chat", data=conversation)

        return None