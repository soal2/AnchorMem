from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Message(StrictModel):
    role: str
    content: str = Field(min_length=1)
    timestamp: int | None = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, value: str) -> str:
        if value not in {"user", "assistant", "system", "tool"}:
            raise ValueError("unsupported role")
        return value


class AddRequest(StrictModel):
    request_id: str = Field(min_length=1)
    messages: list[Message] = Field(min_length=1)
    user_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class AddResponse(StrictModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str


class SearchRequest(StrictModel):
    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)


class Evidence(StrictModel):
    id: str
    content: str
    score: float
    created_at: str | None = None


class SearchResponse(StrictModel):
    data: list[Evidence]
