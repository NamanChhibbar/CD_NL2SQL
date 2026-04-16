"""Exercise vLLM guided decoding against a small set of SQL-generation prompts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from utils.grammar import build_dynamic_grammar, read_grammar_template

DEFAULT_MODEL = "google/gemma-3-12b-it"
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_GRAMMAR_PATH = "guided_decoding/wikisql_grammar.txt"


@dataclass
class Sample:
    """Store one schema, question, and expected SQL example."""

    name: str
    schema: list[str]
    question: str
    expected_sql: str


SAMPLES: list[Sample] = [
    Sample(
        name="trey_johnson_school_club_team",
        schema=["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
        question="What school/club team is Trey Johnson on?",
        expected_sql="SELECT School/Club Team FROM table WHERE Player = Trey Johnson",
    ),
    Sample(
        name="amir_johnson_school_club_team",
        schema=["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
        question="What school/club team is Amir Johnson on?",
        expected_sql="SELECT School/Club Team FROM table WHERE Player = Amir Johnson",
    ),
    Sample(
        name="player_number_21_school",
        schema=["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
        question="What school did player number 21 play for?",
        expected_sql="SELECT School/Club Team FROM table WHERE No. = 21",
    ),
    Sample(
        name="prime_minister_of_italy_take_office",
        schema=[
            "Entered office as Head of State or Government",
            "Began time as senior G8 leader",
            "Ended time as senior G8 leader",
            "Person",
            "Office",
        ],
        question="When did the Prime Minister of Italy take office?",
        expected_sql="SELECT Entered office as Head of State or Government FROM table WHERE Office = Prime Minister of Italy",
    ),
    Sample(
        name="canberra_local_name",
        schema=[
            "Country ( exonym )",
            "Capital ( exonym )",
            "Country ( endonym )",
            "Capital ( endonym )",
            "Official or native language(s) (alphabet/script)",
        ],
        question="What is the local name given to the city of Canberra?",
        expected_sql="SELECT Capital ( endonym ) FROM table WHERE Capital ( exonym ) = Canberra",
    ),
    Sample(
        name="player_number_3_how_many_schools",
        schema=["Player", "No.", "Nationality", "Position", "Years in Toronto", "School/Club Team"],
        question="How many schools did player number 3 play at?",
        expected_sql="SELECT COUNT(School/Club Team) FROM table WHERE No. = 3",
    ),
]


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface for the test script."""
    parser = argparse.ArgumentParser(description="Test vLLM guided decoding with a SQL grammar.")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="vLLM OpenAI-compatible base URL"
    )
    parser.add_argument("--api-key", default="dummy", help="API key value for the OpenAI client")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name served by vLLM")
    parser.add_argument(
        "--grammar-path", default=DEFAULT_GRAMMAR_PATH, help="Path to the grammar file"
    )
    return parser


def build_prompt(schema: list[str], question: str) -> str:
    """Format the user prompt sent to the model for one example."""
    schema_text = ", ".join(f"'{col}'" for col in schema)
    return (
        "Help the user write an SQL statement for their question.\n"
        "You will be given a representation of the SQL database's tables and the users query.\n"
        "Only generate valid SQL output.\n\n"
        f"Table: [{schema_text}]\n\n"
        f"Query: {question}\n\n"
        "SQL Query:\n"
    )


def main() -> None:
    """Run all guided decoding examples and print each model response."""
    parser = build_parser()
    args = parser.parse_args()

    base_grammar = read_grammar_template(Path(args.grammar_path))
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    for i, sample in enumerate(SAMPLES, start=1):
        print("=" * 100)
        print(f"[{i}] {sample.name}")
        print("Question:", sample.question)
        print("Expected SQL:", sample.expected_sql)
        print("Schema:", sample.schema)
        print("-" * 100)

        grammar = build_dynamic_grammar(base_grammar, ["table"], sample.schema)

        prompt = build_prompt(sample.schema, sample.question)

        response = client.responses.create(
            model=args.model,
            input=prompt,
            temperature=0.0,
            max_output_tokens=512,
            extra_body={
                "structured_outputs": {"grammar": grammar},
            },
        )

        output = response.output_text
        print("Model output:")
        print(output.strip())
        print()

    print("=" * 100)
    print("Done.")


if __name__ == "__main__":
    main()
