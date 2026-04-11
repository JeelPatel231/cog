import asyncio
import json
import os
from typing import Any, Optional, Type, cast

from openrouter import OpenRouter
from openrouter.components.responseformatjsonschema import ResponseFormatJSONSchemaTypedDict
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


class OpenRouterChat(ChatProtocol):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "openai/gpt-5.2",
        site_url: str | None = None,
        site_name: str | None = None,
        tool_provider: ToolProvider | None = None,
        max_retries: int = 2,
        retry_delay_seconds: float = 0.75,
    ):
        token = api_key or os.getenv("OPENROUTER_API_KEY")
        if not token:
            raise ValueError("OPENROUTER_API_KEY is required")

        self._client = OpenRouter(
            api_key=token,
            http_referer=site_url,
            x_title=site_name,
        )
        self._model = model
        self._tool_provider = tool_provider
        self._max_retries = max_retries
        self._retry_delay_seconds = retry_delay_seconds

    async def send_message(
        self, message: list[ChatMessage], *, response_format: Optional[Type[BaseModel]] = None
    ) -> AssistantMessage:
        mapped_messages: list[Any] = [self._to_openrouter_message(m) for m in message]
        request_tools: list[dict[str, Any]] = []

        if self._tool_provider is not None:
            definitions = await self._tool_provider.get_tool_definitions()
            request_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": definition["name"],
                        "description": definition["description"],
                        "parameters": definition["input_schema"],
                    },
                }
                for definition in definitions
            ]

            # Put all tool definitions in a system prompt only on the very first message.
            if len(message) == 1 and definitions:
                system_prompt = await self._tool_provider.get_system_prompt()
                mapped_messages.insert(0, {"role": "system", "content": system_prompt})

        completion: Any = None
        last_error: Exception | None = None

        response_format_dict: ResponseFormatJSONSchemaTypedDict | None = None
        if response_format is not None:
            response_format_dict = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_format.__name__,
                    "schema_": response_format.model_json_schema(),
                    "strict": True,
                },
            }

        completion = await self._client.chat.send_async(
            model=self._model,
            messages=mapped_messages,
            tools=cast(Any, request_tools if request_tools else None),
            stream=False,
            response_format=response_format_dict,
        )

        choice = completion.choices[0].message

        tool_calls = []
        if choice.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tool_call.id,
                    name=getattr(getattr(tool_call, "function", None), "name", ""),
                    arguments=json.loads(
                        getattr(getattr(tool_call, "function", None), "arguments", "{}")
                    ),
                )
                for tool_call in choice.tool_calls
            ]

        return AssistantMessage(
            role="assistant",
            content=TextMessageContent(text=self._extract_text(choice.content)),
            tool_calls=tool_calls,
        )

    def _to_openrouter_message(self, message: ChatMessage) -> dict[str, Any]:
        if isinstance(message, AssistantMessage):
            role = "assistant"
            content = message.content
            if isinstance(content, TextMessageContent):
                return {
                    "role": role,
                    "content": content.text,
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments),
                            },
                        }
                        for tool_call in message.tool_calls
                    ],
                }

            if isinstance(content, ImageMessageContent):
                return {
                    "role": role,
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{content.base64}",
                            },
                        }
                    ],
                }
            raise ValueError(
                f"Unsupported content type for ToolResponseMessage: {type(content)!r}"
            )

        elif isinstance(message, UserMessage):
            role = "user"
            content = message.content
            if isinstance(content, TextMessageContent):
                return {"role": role, "content": content.text}

            if isinstance(content, ImageMessageContent):
                return {
                    "role": role,
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{content.base64}",
                            },
                        }
                    ],
                }

            raise ValueError(
                f"Unsupported content type for ToolResponseMessage: {type(content)!r}"
            )

        elif isinstance(message, ToolResponseMessage):
            role = "tool"
            content = message.content
            if isinstance(content, TextMessageContent):
                return {
                    "role": role,
                    "content": content.text,
                    "tool_call_id": message.id,
                    "name": message.name,
                }
            raise ValueError(
                f"Unsupported content type for ToolResponseMessage: {type(content)!r}"
            )

        elif isinstance(message, SystemMessage):
            content = message.content
            if isinstance(content, TextMessageContent):
                return {"role": "system", "content": content.text}
            raise ValueError(
                f"Unsupported content type for SystemMessage: {type(content)!r}"
            )

        raise ValueError(f"Unsupported message role: {message.role}")

    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                item_type = getattr(item, "type", None)
                if item_type is None and isinstance(item, dict):
                    item_type = item.get("type")

                if item_type == "text":
                    text_value = getattr(item, "text", None)
                    if text_value is None and isinstance(item, dict):
                        text_value = item.get("text")
                    if isinstance(text_value, str):
                        text_parts.append(text_value)

            return "\n".join(text_parts)

        return str(content)
