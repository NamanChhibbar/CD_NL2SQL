"""Generate model outputs for NL2SQL datasets."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import logging
import os
from pathlib import Path
import re
import sqlite3
import time
from typing import Any

from datasets import Dataset
from openai import APIError, APITimeoutError, OpenAI
from tqdm import tqdm

from utils.data import get_data
from utils.enums import DatasetNames, GemmaModels
from utils.grammar import build_dynamic_grammar, extract_schema_info, read_grammar_template
from utils.models import ChatbotMetadata, ChatbotOutput, QueryDetails
from utils.prompts import SQALE_PROMPT, WIKISQL_PROMPT

API_KEY = os.getenv("NL2SQL_API_KEY", "dummy")
LOGGER = logging.getLogger(__name__)

DEFAULT_GRAMMAR_PATH = "guided_decoding/sql_grammar.txt"
SYSTEM_PROMPT = "You are a helpful assistant that generates SQL queries."
AGENT_CRITIC_SYSTEM_PROMPT = (
    "You generate exactly one SQL query that answers the user's question. "
    "Return SQL only. Do not include markdown fences, explanations, or commentary."
)
SQL_FENCE_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
SQLITE_SYNTAX_ERROR_MARKERS = (
    "syntax error",
    "incomplete input",
    "unrecognized token",
    "unterminated",
    "unexpected",
)


def build_prompt_and_reference(
    dataset_name: DatasetNames, item: dict[str, Any]
) -> tuple[str, str, str, str]:
    """Build the prompt and reference fields for one dataset row."""
    if dataset_name == DatasetNames.WIKISQL:
        prompt_template = WIKISQL_PROMPT
        table = str(item["table"]["header"])
        query = str(item["question"])
        human_sql = str(item["sql"]["human_readable"])
    elif dataset_name == DatasetNames.SQALE:
        prompt_template = SQALE_PROMPT
        table = str(item["schema"])
        query = str(item["question"])
        human_sql = str(item["query"])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return prompt_template.format(table=table, query=query), table, query, human_sql


def extract_sql_from_response(text: str) -> str:
    """Normalize model output down to SQL text."""
    stripped = text.strip()
    if not stripped:
        return stripped

    fence_match = SQL_FENCE_RE.search(stripped)
    if fence_match:
        return fence_match.group(1).strip()

    return stripped


def is_sqlite_syntax_error(message: str) -> bool:
    """Return True when SQLite indicates a parse-level issue."""
    lowered = message.lower()
    return any(marker in lowered for marker in SQLITE_SYNTAX_ERROR_MARKERS)


def validate_sql_with_sqlite(sql_text: str) -> tuple[bool, str | None]:
    """Use SQLite parsing to catch syntax errors in generated SQL."""
    sql = extract_sql_from_response(sql_text)
    if not sql:
        return False, "The response was empty."

    try:
        with sqlite3.connect(":memory:") as connection:
            connection.execute(f"EXPLAIN QUERY PLAN {sql}")
    except sqlite3.Error as exc:
        message = str(exc)
        if is_sqlite_syntax_error(message):
            return False, message
        return True, None

    return True, None


def create_response(
    client: OpenAI,
    model_name: str,
    *,
    instructions: str,
    input_payload: str | list[dict[str, str]],
    base_grammar: str | None,
    item: dict[str, Any] | None,
    dataset_name: DatasetNames | None,
    max_completion_tokens: int,
) -> str:
    """Send one request to the Responses API and return text output."""
    request_kwargs: dict[str, Any] = {
        "model": model_name,
        "instructions": instructions,
        "input": input_payload,
        "max_output_tokens": max_completion_tokens,
    }

    if base_grammar is not None:
        if item is None or dataset_name is None:
            raise ValueError("Grammar-constrained generation requires the dataset item and name.")
        request_kwargs["extra_body"] = {
            "structured_outputs": {
                "grammar": build_dynamic_grammar(
                    base_grammar, *extract_schema_info(item, dataset_name)
                )
            }
        }
        request_kwargs["temperature"] = 0.0
        request_kwargs["max_output_tokens"] = max_completion_tokens

    response = client.responses.create(**request_kwargs)
    return response.output_text.strip()


def generate_single_pass(
    *,
    client: OpenAI,
    model_name: str,
    prompt: str,
    dataset_name: DatasetNames,
    item: dict[str, Any],
    base_grammar: str | None,
    max_completion_tokens: int,
) -> tuple[str, int, str | None]:
    """Run a single model call."""
    output_text = create_response(
        client,
        model_name,
        instructions=SYSTEM_PROMPT,
        input_payload=prompt,
        base_grammar=base_grammar,
        item=item,
        dataset_name=dataset_name,
        max_completion_tokens=max_completion_tokens,
    )
    return output_text, 1, None


def generate_with_agent_critic(
    *,
    client: OpenAI,
    model_name: str,
    prompt: str,
    max_completion_tokens: int,
    max_rounds: int,
) -> tuple[str, int, str | None]:
    """Run an agent-critic loop using SQLite syntax validation as the critic."""
    conversation: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    last_output = ""
    last_error: str | None = None

    for round_index in range(1, max_rounds + 1):
        last_output = create_response(
            client,
            model_name,
            instructions=AGENT_CRITIC_SYSTEM_PROMPT,
            input_payload=conversation,
            base_grammar=None,
            item=None,
            dataset_name=None,
            max_completion_tokens=max_completion_tokens,
        )

        is_valid, error_message = validate_sql_with_sqlite(last_output)
        if is_valid:
            return last_output, round_index, None

        last_error = error_message
        conversation.extend(
            [
                {"role": "assistant", "content": last_output},
                {
                    "role": "user",
                    "content": (
                        "The SQL you produced is not syntactically valid for SQLite.\n"
                        f"SQLite reported: {error_message}\n"
                        "Generate a corrected SQL query only."
                    ),
                },
            ]
        )

    return last_output, max_rounds, last_error


def process_item(
    item: dict[str, Any],
    client: OpenAI,
    model_name: str,
    dataset_name: DatasetNames,
    *,
    base_grammar: str | None = None,
    use_agent_critic: bool = False,
    max_completion_tokens: int = 256,
    agent_critic_rounds: int = 3,
    max_retries: int = 3,
) -> ChatbotOutput | None:
    """Process one dataset item and return the serialized output record."""
    prompt, table, query, human_sql = build_prompt_and_reference(dataset_name, item)

    for attempt in range(max_retries):
        try:
            if use_agent_critic:
                response_text, rounds_used, validation_error = generate_with_agent_critic(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    max_completion_tokens=max_completion_tokens,
                    max_rounds=agent_critic_rounds,
                )
                generation_approach = "agent_critic"
            else:
                response_text, rounds_used, validation_error = generate_single_pass(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    dataset_name=dataset_name,
                    item=item,
                    base_grammar=base_grammar,
                    max_completion_tokens=max_completion_tokens,
                )
                generation_approach = (
                    "guided_decoding" if base_grammar is not None else "single_pass"
                )

            return ChatbotOutput(
                prompt=prompt,
                response=response_text,
                human_sql=human_sql,
                metadata=ChatbotMetadata(
                    model_name=model_name,
                    used_guided_decoding=base_grammar is not None,
                    generation_approach=generation_approach,
                    agent_critic_rounds=rounds_used if use_agent_critic else None,
                    final_validation_error=validation_error,
                ),
                query_details=QueryDetails(
                    dataset_name=str(dataset_name),
                    raw_question=query,
                    schema_or_table_details=table,
                ),
            )
        except (APIError, APITimeoutError) as exc:
            LOGGER.error(
                "Error processing item (attempt %s/%s): %s",
                attempt + 1,
                max_retries,
                exc,
            )
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
            else:
                LOGGER.error("Max retries reached. Skipping item.")
                return None

    return None


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-split",
        type=str,
        choices=["train", "validation", "test"],
        default="validation",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[str(v) for v in DatasetNames],
        default=str(DatasetNames.WIKISQL),
        help="Dataset name to use",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=[str(v) for v in GemmaModels],
        default=str(GemmaModels.GEMMA3_270M),
        help="Model name to use",
    )
    parser.add_argument(
        "--endpoint",
        dest="endpoints",
        action="append",
        type=str,
        help="Model endpoint to use. Can be specified multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save outputs",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=12,
        help="Number of jobs to use for parallel processing",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        default=False,
        help="Enable guided decoding with an EBNF grammar constraint",
    )
    parser.add_argument(
        "--agent-critic",
        action="store_true",
        default=False,
        help="Enable an agent-critic loop where SQLite syntax validation critiques bad SQL",
    )
    parser.add_argument(
        "--agent-critic-rounds",
        type=int,
        default=3,
        help="Maximum rounds for the agent-critic loop",
    )
    parser.add_argument(
        "--grammar-path",
        type=Path,
        default=DEFAULT_GRAMMAR_PATH,
        help="Path to the EBNF grammar template (used with --guided-decoding)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum tokens in the model completion",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for API requests",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        help="Limit the number of dataset rows processed",
    )
    return parser


def main() -> None:
    """Entry point for output generation."""
    parser = build_parser()
    args = parser.parse_args()

    if args.guided_decoding and args.agent_critic:
        parser.error("--guided-decoding and --agent-critic cannot be used together.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = DatasetNames(args.dataset_name)
    dataset: Dataset = get_data(dataset_name, args.dataset_split)

    if args.max_items is not None:
        dataset = dataset.select(range(min(args.max_items, len(dataset))))

    if not args.endpoints:
        parser.error("At least one --endpoint must be provided.")

    base_grammar: str | None = None
    if args.guided_decoding:
        base_grammar = read_grammar_template(Path(args.grammar_path))
        print(f"Guided decoding enabled (grammar: {args.grammar_path})")
    elif args.agent_critic:
        print(f"Agent-critic enabled (max rounds: {args.agent_critic_rounds})")

    clients = [
        OpenAI(base_url=endpoint, api_key=API_KEY, timeout=args.timeout)
        for endpoint in args.endpoints
    ]
    client_cycle = itertools.cycle(clients)

    suffix = ""
    if args.guided_decoding:
        suffix = "_guided"
    elif args.agent_critic:
        suffix = "_agent_critic"

    output_file_path = output_dir / (
        f"{args.model_name.replace('/', '-')}_{args.dataset_name}"
        f"_{args.dataset_split}{suffix}.jsonl"
    )

    print(f"Processing {len(dataset)} items...")
    with (
        output_file_path.open("w", encoding="utf-8") as output_file,
        ThreadPoolExecutor(max_workers=args.num_jobs) as executor,
    ):
        futures = [
            executor.submit(
                process_item,
                item,
                next(client_cycle),
                args.model_name,
                dataset_name,
                base_grammar=base_grammar,
                use_agent_critic=args.agent_critic,
                max_completion_tokens=args.max_completion_tokens,
                agent_critic_rounds=args.agent_critic_rounds,
            )
            for item in dataset
        ]

        processed_count = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating outputs"):
            result = future.result()
            if result is None:
                continue
            output_file.write(result.model_dump_json() + "\n")
            output_file.flush()
            processed_count += 1

    print(f"Processed {processed_count} items. Results saved to {output_file_path}")


if __name__ == "__main__":
    main()
