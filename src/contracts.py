from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .temporal import time_bounds


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
    project_id: str | None = None

    @field_validator("project_id")
    @classmethod
    def normalize_project_id(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None


class AddResponse(StrictModel):
    success: bool = True
    request_id: str
    user_id: str
    session_id: str
    unresolved_entry_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class EventTimeRange(StrictModel):
    start: str | None = None
    end: str | None = None

    @field_validator("start", "end")
    @classmethod
    def validate_boundary(cls, value: str | None) -> str | None:
        value = value.strip() or None if value is not None else None
        if value is not None:
            time_bounds(value)
        return value

    @model_validator(mode="after")
    def ordered_range(self):
        if self.start is not None and self.end is not None:
            start, end = time_bounds(self.start)[0], time_bounds(self.end)[0]
            if (start.utcoffset() is None) != (end.utcoffset() is None):
                raise ValueError("event_time_range boundaries require comparable timezones")
            if start >= end:
                raise ValueError("event_time_range start must precede end")
        return self


class SearchRequest(StrictModel):
    query: str = Field(min_length=1)
    options: list[str] | None = None
    user_id: str = Field(min_length=1)
    top_k: int = Field(ge=1)
    project_id: str | None = None
    event_time_range: EventTimeRange | None = None

    @field_validator("project_id")
    @classmethod
    def normalize_project_id(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("event_time_range")
    @classmethod
    def normalize_time_range(cls, value: EventTimeRange | None) -> EventTimeRange | None:
        return value if value is not None and (value.start is not None or value.end is not None) else None


class Evidence(StrictModel):
    id: str
    content: str
    score: float
    created_at: str | None = None
    entry_ids: list[str] = Field(default_factory=list)
    topic: str
    scope: str
    project_id: str | None = None
    requirement_priority: int | None = None


class SearchResponse(StrictModel):
    data: list[Evidence]
