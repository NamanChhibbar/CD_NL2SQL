"""Pydantic Models for storing output results."""

from pathlib import Path

from pydantic import BaseModel


class ChatbotMetadata(BaseModel):
    """Metadata for the chatbot."""

    model_name: str
    used_guided_decoding: bool
    generation_approach: str | None = None
    agent_critic_rounds: int | None = None
    final_validation_error: str | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    response_status: str | None = None
    incomplete_reason: str | None = None
    output_tokens: int | None = None
    guided_decoding_grammar_path: Path | str | None = None


class QueryDetails(BaseModel):
    """Dataset Input Details."""

    dataset_name: str
    dataset_index: int | None = None
    raw_question: str
    schema_or_table_details: str


class ChatbotOutput(BaseModel):
    """Output model for the chatbot."""

    prompt: str
    response: str
    human_sql: str
    metadata: ChatbotMetadata
    query_details: QueryDetails
