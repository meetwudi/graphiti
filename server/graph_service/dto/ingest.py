from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from graph_service.dto.common import Message


class AddMessagesBulkRequest(BaseModel):
    request_id: str = Field(..., min_length=1, description='Idempotency identity for this batch')
    group_id: str = Field(
        ...,
        min_length=1,
        pattern=r'^[a-zA-Z0-9_-]+$',
        description='The graph identifier of the messages to add',
    )
    messages: list[Message] = Field(
        ..., min_length=1, description='The messages to add in one bulk operation'
    )

    @model_validator(mode='after')
    def require_unique_uuids(self) -> Self:
        uuids = [message.uuid for message in self.messages]
        if any(uuid is None for uuid in uuids):
            raise ValueError('bulk messages must include deterministic episode UUIDs')
        if len(set(uuids)) != len(uuids):
            raise ValueError('bulk message episode UUIDs must be unique')
        return self


class AddMessagesBulkResponse(BaseModel):
    success: bool
    request_id: str
    message: str
    episode_count: int
    episode_uuids: list[str]
    processed_episode_uuids: list[str]
