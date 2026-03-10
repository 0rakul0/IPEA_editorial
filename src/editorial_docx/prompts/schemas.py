from __future__ import annotations

from pydantic import BaseModel, ConfigDict, RootModel


class AgentCommentPayload(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    category: str
    message: str
    paragraph_index: int | None = None
    issue_excerpt: str = ""
    suggested_fix: str = ""


class AgentCommentsPayload(RootModel[list[AgentCommentPayload]]):
    pass


class PromptProfile(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: str
    description: str
    instruction: str