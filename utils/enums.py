"""Enums used across the project."""

from enum import StrEnum


class GemmaModels(StrEnum):
    """Enum for supported models in the Gemma framework."""

    GEMMA3_270M = "google/gemma-3-270m-it"
    GEMMA3_1B = "google/gemma-3-1b-it"
    GEMMA3_4B = "google/gemma-3-4b-it"
    GEMMA3_12B = "google/gemma-3-12b-it"
    GEMMA3_27B = "google/gemma-3-27b-it"


class DatasetNames(StrEnum):
    """Enum for supported datasets."""

    SQALE = "SQaLe"
    WIKISQL = "wikisql"
