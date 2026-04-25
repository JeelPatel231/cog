import json
import os
import asyncio
from typing import Any, Iterable, Optional, Type, cast

from openai import AsyncOpenAI, omit, Omit
from openai.types.chat import ChatCompletionToolUnionParam
from openai.types.chat.chat_completion_message_function_tool_call import Function
from openai.types.shared_params.response_format_json_schema import (
    ResponseFormatJSONSchema,
)
from pydantic import BaseModel

from core.chat import (
    AssistantMessage,
    ChatMessage,
    ChatProtocol,
    ImageMessageContent,
    SystemMessage,
    TextMessageContent,
    ToolCall,
    ToolResponseMessage,
    UserMessage,
)
from core.tool_provider import ToolProvider
from core.logger import logger


class OpenAIChat(ChatProtocol):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "gpt-4.1",
        tool_provider: ToolProvider | None = None,
    ):
        token = api_key or os.getenv("OPENAI_API_KEY")
        if not token:
            raise ValueError("OPENAI_API_KEY is required")

        self._client = AsyncOpenAI(base_url=base_url, api_key=token)
        self._model = model
        self._tool_provider = tool_provider

    async def send_message(
        self,
        message: list[ChatMessage],
        *,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> AssistantMessage:
        mapped_messages = [self._to_openai_message(m) for m in message]

        tools: Iterable[ChatCompletionToolUnionParam] | Omit = omit
        if self._tool_provider:
            definitions = await self._tool_provider.get_tool_definitions()
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": d["name"],
                        "description": d["description"],
                        "parameters": d["input_schema"],
                    },
                }
                for d in definitions
            ]

        response_format_dict: ResponseFormatJSONSchema | Omit = omit
        if response_format:
            response_format_dict = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema": response_format.model_json_schema(),
                    "strict": True,
                },
            }

        completion = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(Any, mapped_messages),
            tools=tools,
            response_format=response_format_dict,
        )

        if not completion.choices:
            raise ValueError("No choices returned from OpenAI")

        choice = completion.choices[0].message

        if choice is None:
            raise ValueError("OpenAI returned a choice with no message")

        tool_calls = []
        if choice.tool_calls is not None:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments or "{}"),
                )
                for tc in choice.tool_calls
                if tc.type == 'function'
            ]
        return AssistantMessage(
            role="assistant",
            content=TextMessageContent(text=self._extract_text(choice.content)),
            tool_calls=tool_calls,
        )

    def _to_openai_message(self, message: ChatMessage) -> dict[str, Any]:
        if isinstance(message, AssistantMessage):
            if isinstance(message.content, TextMessageContent):
                return {
                    "role": "assistant",
                    "content": message.content.text,
                    "tool_calls": (
                        [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in message.tool_calls
                        ]
                        if message.tool_calls
                        else None
                    ),
                }

            if isinstance(message.content, ImageMessageContent):
                return {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{message.content.base64}"
                            },
                        }
                    ],
                }

        elif isinstance(message, UserMessage):
            if isinstance(message.content, TextMessageContent):
                return {"role": "user", "content": message.content.text}

            if isinstance(message.content, ImageMessageContent):
                return {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{message.content.base64}"
                            },
                        }
                    ],
                }

        elif isinstance(message, ToolResponseMessage):
            return {
                "role": "tool",
                "tool_call_id": message.id,
                "name": message.name,
                "content": message.content.text,
            }

        elif isinstance(message, SystemMessage):
            return {
                "role": "system",
                "content": message.content.text,
            }

        raise ValueError(f"Unsupported message type: {type(message)}")

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            return "\n".join(parts)

        return str(content)
