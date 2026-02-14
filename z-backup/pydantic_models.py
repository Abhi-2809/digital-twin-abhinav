"""Pydantic models for Abhinav Digital Twin RAG system."""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# -- Router models --

class RouterOutput(BaseModel):
    """Validated output from the query router LLM."""

    collection: Literal["personal", "professional", "both"] = Field(
        description="Target collection(s) to search"
    )
    type: Literal["fact", "comprehensive"] = Field(
        description="Query type: fact (k=3) or comprehensive (k=10)"
    )
    k: Literal[3, 10] = Field(
        description="Number of documents to retrieve per collection"
    )
    reasoning: str = Field(
        default="",
        description="Brief one-sentence explanation for the routing decision"
    )

    @classmethod
    def from_llm_response(cls, data: dict) -> "RouterOutput":
        """Build and validate RouterOutput from parsed LLM response."""
        collection = data.get("collection", "both")
        if collection not in ("personal", "professional", "both"):
            collection = "both"
        query_type = data.get("type", "comprehensive")
        if query_type not in ("fact", "comprehensive"):
            query_type = "comprehensive"
        k = data.get("k", 10 if query_type == "comprehensive" else 3)
        if k not in (3, 10):
            k = 3 if query_type == "fact" else 10
        reasoning = data.get("reasoning", "")
        if not isinstance(reasoning, str):
            reasoning = str(reasoning) if reasoning else ""
        return cls(
            collection=collection,
            type=query_type,
            k=k,
            reasoning=reasoning.strip()
        )


# -- API request / response models --

class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for /api/chat."""
    query: str = Field(..., min_length=1, description="User query")
    stream: bool = Field(default=True, description="Stream the response token-by-token")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous conversation messages for context"
    )


class SourceInfo(BaseModel):
    """A single source reference."""
    filename: str
    page: int | str
    category: str


class RoutingInfo(BaseModel):
    """Routing metadata returned alongside a chat response."""
    collection: str
    query_type: str
    k: int
    reasoning: str


class ChatResponse(BaseModel):
    """Full (non-streamed) response for /api/chat."""
    answer: str
    routing: RoutingInfo
    sources: list[SourceInfo]
    num_documents: int


class HealthResponse(BaseModel):
    """Response for /health."""
    status: str = "ok"
    collections: list[str] = []
