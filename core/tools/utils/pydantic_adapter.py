from typing import Self

from pydantic import BaseModel

class PydanticToolArgs(BaseModel):
    @classmethod
    def tool_json_schema(cls) -> dict:
        return cls.model_json_schema()
    
    @classmethod
    def tool_validate(cls, arguments: dict) -> Self:
        return cls.model_validate(arguments)