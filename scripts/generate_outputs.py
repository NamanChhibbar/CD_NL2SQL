"""Generate outputs using either the Eval or Test datasets and a given model endpoint."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import logging
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
from openai import OpenAI
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


def process_item(
    item: dict[str, Any],
    client: OpenAI,
    model_name: str,
    dataset_name: DatasetNames,
    *,
    base_grammar: str | None = None,
    max_completion_tokens: int = 256,
) -> ChatbotOutput:
    """Process a single dataset item and return the chatbot output.

    When *base_grammar* is provided the request goes through the
    ``chat.completions`` endpoint with a per-item EBNF grammar constraint
    (guided decoding).  Otherwise the ``responses`` endpoint is used without
    any grammar constraint.
    """
    if dataset_name == DatasetNames.WIKISQL:
        prompt_template = WIKISQL_PROMPT
        table = item["table"]["header"]
        query = item["question"]
        human_sql = item["sql"]["human_readable"]
    else:
        prompt_template = SQALE_PROMPT
        table = item["schema"]
        query = item["question"]
        human_sql = item["query"]

    prompt = prompt_template.format(table=table, query=query)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        extra_body={
            "structured_outputs": {
                "grammar": build_dynamic_grammar(
                    base_grammar, *extract_schema_info(item, dataset_name)
                )
            },
        }
        if base_grammar is not None
        else {},
        temperature=0.0 if base_grammar is not None else None,
        max_output_tokens=max_completion_tokens if base_grammar is not None else None,
    )
    response_text = response.output_text

    return ChatbotOutput(
        prompt=prompt,
        response=response_text,
        human_sql=human_sql,
        metadata=ChatbotMetadata(
            model_name=model_name,
            used_guided_decoding=base_grammar is not None,
        ),
        query_details=QueryDetails(
            dataset_name=str(dataset_name),
            raw_question=query,
            schema_or_table_details=str(table),
        ),
    )


def main() -> None:
    """Entry point of script."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
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
        default=GemmaModels.GEMMA3_270M,
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
        "--grammar-path",
        type=Path,
        default=DEFAULT_GRAMMAR_PATH,
        help="Path to the EBNF grammar template (used with --guided-decoding)",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=512,
        help="Maximum tokens in the model completion (used with --guided-decoding)",
    )
    args = parser.parse_args()
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset: Dataset = get_data(DatasetNames(args.dataset_name), args.dataset_split)

    if not args.endpoints:
        print("No endpoints provided. Exiting.")
        return

    base_grammar: str | None = None
    if args.guided_decoding:
        base_grammar = read_grammar_template(Path(args.grammar_path))
        print(f"Guided decoding enabled (grammar: {args.grammar_path})")

    clients = [
        OpenAI(base_url=endpoint, api_key=API_KEY, timeout=600) for endpoint in args.endpoints
    ]
    client_cycle = itertools.cycle(clients)

    results: list[ChatbotOutput] = []

    print(f"Processing {len(dataset)} items...")
    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                next(client_cycle),
                args.model_name,
                DatasetNames(args.dataset_name),
                base_grammar=base_grammar,
                max_completion_tokens=args.max_completion_tokens,
            )
            for item in dataset
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating outputs"):
            results.append(future.result())

    print(f"Processed {len(results)} items.")

    suffix = "_guided" if args.guided_decoding else ""
    filename = (
        f"{args.model_name.replace('/', '-')}_{args.dataset_name}"
        f"_{args.dataset_split}{suffix}.jsonl"
    )
    with open(
        output_dir / filename,
        "w",
        encoding="utf-8",
    ) as processed_response_file:
        for response in results:
            processed_response_file.write(response.model_dump_json())
            processed_response_file.write("\n")


if __name__ == "__main__":
    main()
