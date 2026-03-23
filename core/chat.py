from typing import Any, Literal, Protocol
from pydantic import BaseModel

class TextMessageContent(BaseModel):
    text: str

class ImageMessageContent(BaseModel):
    base64: str

MessageContent = TextMessageContent | ImageMessageContent

class UserMessage(BaseModel):
    role: Literal['user']
    content: MessageContent

class AssistantMessage(BaseModel):
    role: Literal['assistant']
    content: MessageContent
    tool_calls: list[tuple[str, dict[str, Any]]] = []

class ToolResponseMessage(BaseModel):
    role: Literal['tool_call']
    content: MessageContent

ChatMessage = UserMessage | AssistantMessage | ToolResponseMessage

class ChatProtocol(Protocol):
    """ 
    this interface defines the contract for a chat agent. 
    it has a single method, send_message, which takes a list of chat messages and returns an assistant message.

    the concrete implementation may use anything to process and generate a response such as api calls.
    its the responsibility of this class to map the input messages to the format required by the underlying implementation, 
    and to map the output of the implementation back to an AssistantMessage.
    """
    async def send_message(self, message: list[ChatMessage]) -> AssistantMessage: ...
