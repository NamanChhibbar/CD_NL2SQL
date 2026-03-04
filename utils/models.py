"""Pydantic Models for storing output results."""

from pydantic import BaseModel


class ChatbotMetadata(BaseModel):
    """Metadata for the chatbot."""

    model_name: str
    used_guided_decoding: bool


class QueryDetails(BaseModel):
    """Dataset Input Details."""

    dataset_name: str
    raw_question: str
    schema_or_table_details: str


class ChatbotOutput(BaseModel):
    """Output model for the chatbot."""

    prompt: str
    response: str
    metadata: ChatbotMetadata
    query_details: QueryDetails
