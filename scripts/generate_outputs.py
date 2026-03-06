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
from utils.models import ChatbotMetadata, ChatbotOutput, QueryDetails
from utils.prompts import SQALE_PROMPT, WIKISQL_PROMPT

API_KEY = os.getenv("NL2SQL_API_KEY", "dummy")
LOGGER = logging.getLogger(__name__)


def process_item(
    item: dict[str, Any],
    client: OpenAI,
    model_name: str,
    dataset_name: DatasetNames,
) -> ChatbotOutput:
    """Process a single dataset item and return the chatbot output."""
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
    )

    return ChatbotOutput(
        prompt=prompt,
        response=response.output_text,
        human_sql=human_sql,
        metadata=ChatbotMetadata(model_name=model_name, used_guided_decoding=False),
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
    args = parser.parse_args()
    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset: Dataset = get_data(DatasetNames(args.dataset_name), args.dataset_split)

    if not args.endpoints:
        print("No endpoints provided. Exiting.")
        return

    clients = [OpenAI(base_url=endpoint, api_key=API_KEY) for endpoint in args.endpoints]
    client_cycle = itertools.cycle(clients)

    results: list[ChatbotOutput] = []

    print(f"Processing {len(dataset)} items...")
    # Example of using ThreadPoolExecutor to process dataset items in parallel
    with ThreadPoolExecutor(max_workers=args.num_jobs) as executor:
        futures = [
            executor.submit(
                process_item,
                item,
                next(client_cycle),
                args.model_name,
                DatasetNames(args.dataset_name),
            )
            for item in dataset
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating outputs"):
            results.append(future.result())

    print(f"Processed {len(results)} items.")

    with open(
        output_dir
        / f"{args.model_name.replace('/', '-')}_{args.dataset_name}_{args.dataset_split}.jsonl",
        "w",
        encoding="utf-8",
    ) as processed_response_file:
        for response in results:
            processed_response_file.write(response.model_dump_json())
            processed_response_file.write("\n")


if __name__ == "__main__":
    main()
