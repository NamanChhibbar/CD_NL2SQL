"""Contains utility functions for data processing."""

from typing import Literal

from anyio.functools import lru_cache
from datasets import Dataset, load_dataset

from utils.enums import DatasetNames


@lru_cache(maxsize=None)
def split_data(
    dataset: Dataset, train_frac=0.8, val_frac=0.1, random_seed=42
) -> tuple[Dataset, Dataset, Dataset]:
    """Splits a dataset from Hugging Face into train, validation, and test sets based on the specified sizes.

    Parameters:
      dataset (Dataset): The input dataset to be split.
      train_frac (float | 0.8): The proportion of the dataset to be used for training.
      val_frac (float | 0.1): The proportion of the dataset to be used for validation.
      random_seed (int | 42): The seed for random shuffling to ensure reproducibility.

    Returns:
      tuple[Dataset, Dataset, Dataset]: A tuple containing the train, validation, and test datasets.
    """
    dataset.shuffle(seed=random_seed)
    size = dataset.num_rows
    train_frac = int(size * train_frac)
    val_frac = int(size * val_frac)
    train_data = dataset.select(range(train_frac))
    val_data = dataset.select(range(train_frac, train_frac + val_frac))
    test_data = dataset.select(range(train_frac + val_frac, size))
    return train_data, val_data, test_data


def get_data(dataset_name: DatasetNames, split: Literal["train", "validation", "test"]) -> Dataset:
    """Get a dataset from the specified dataset name and split."""
    match dataset_name:
        case DatasetNames.SQALE:
            train, val, test = split_data(
                load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", trust_remote_code=True)["train"],
                0.8,
                0.1,
                random_seed=42,
            )
            match split:
                case "train":
                    return train
                case "validation":
                    return val
                case "test":
                    return test
        case DatasetNames.WIKISQL:
            return load_dataset("wikisql")[split]
        case _:
            raise ValueError(f"Invalid dataset name: {dataset_name}")
