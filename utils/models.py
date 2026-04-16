"""Pydantic Models for storing output results."""

from pydantic import BaseModel


class ChatbotMetadata(BaseModel):
    """Metadata for the chatbot."""

    model_name: str
    used_guided_decoding: bool
    generation_approach: str | None = None
    agent_critic_rounds: int | None = None
    final_validation_error: str | None = None


class QueryDetails(BaseModel):
    """Dataset Input Details."""

    dataset_name: str
    raw_question: str
    schema_or_table_details: str


class ChatbotOutput(BaseModel):
    """Output model for the chatbot."""

    prompt: str
    response: str
    human_sql: str
    metadata: ChatbotMetadata
    query_details: QueryDetails
