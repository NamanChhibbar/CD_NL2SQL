"""Fine-tune a Gemma model using Unsloth and LoRA."""

from unsloth import FastVisionModel

import argparse
from pathlib import Path

from datasets import Dataset
import torch
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, TrainingArguments
from trl import SFTTrainer

from utils.data import get_data
from utils.enums import DatasetNames, GemmaModels
from utils.models import ChatbotMetadata, ChatbotOutput, QueryDetails
from utils.prompts import SQALE_PROMPT, WIKISQL_PROMPT


def build_prompt_and_reference(dataset_name: DatasetNames, item: dict) -> tuple[str, str, str, str]:
    """Build the generation prompt and extract reference fields for one example."""
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


def format_dataset(dataset_name: DatasetNames, dataset: Dataset, tokenizer):
    """Format the dataset according to the specified dataset name."""
    if dataset_name == DatasetNames.WIKISQL:
        prompt_template = WIKISQL_PROMPT

        def formatting_prompts_func(examples):
            instructions = [
                prompt_template.format(table=t["header"], query=q)
                for t, q in zip(examples["table"], examples["question"], strict=True)
            ]
            outputs = [s["human_readable"] for s in examples["sql"]]
            texts = [
                f"{instruction}{output}{tokenizer.eos_token}"
                for instruction, output in zip(instructions, outputs, strict=True)
            ]
            return {"text": texts}

    elif dataset_name == DatasetNames.SQALE:
        prompt_template = SQALE_PROMPT

        def formatting_prompts_func(examples):
            instructions = [
                prompt_template.format(table=s, query=q)
                for s, q in zip(examples["schema"], examples["question"], strict=True)
            ]
            outputs = examples["query"]
            texts = [
                f"{instruction}{output}{tokenizer.eos_token}"
                for instruction, output in zip(instructions, outputs, strict=True)
            ]
            return {"text": texts}
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset.map(formatting_prompts_func, batched=True)


def generate_validation_outputs(
    model,
    tokenizer,
    dataset_name: DatasetNames,
    model_output_dir: Path,
    output_path: Path,
    *,
    max_new_tokens: int,
) -> None:
    """Generate SQL predictions on the validation split for the trained model."""
    validation_dataset = get_data(dataset_name, "validation")

    FastVisionModel.for_inference(model)
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as output_file:
        for item in tqdm(validation_dataset, desc="Generating validation outputs"):
            prompt, table, query, human_sql = build_prompt_and_reference(dataset_name, item)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            prompt_length = inputs["input_ids"].shape[1]
            response = tokenizer.decode(
                generated[0][prompt_length:],
                skip_special_tokens=True,
            ).strip()

            output = ChatbotOutput(
                prompt=prompt,
                response=response,
                human_sql=human_sql,
                metadata=ChatbotMetadata(
                    model_name=str(model_output_dir),
                    used_guided_decoding=False,
                ),
                query_details=QueryDetails(
                    dataset_name=str(dataset_name),
                    raw_question=query,
                    schema_or_table_details=table,
                ),
            )
            output_file.write(output.model_dump_json() + "\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        choices=[str(v) for v in GemmaModels],
        default=GemmaModels.GEMMA3_4B,
        help="Gemma model to fine-tune",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        choices=[str(v) for v in DatasetNames],
        default=str(DatasetNames.WIKISQL),
        help="Dataset name to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to save the fine-tuned model",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        default=True,
        help="Load the model in 4-bit quantization",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="LoRA rank",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size per device",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Warmup steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Logging steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=3407,
        help="Random seed",
    )
    parser.add_argument(
        "--validation-max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate per validation example after training",
    )
    parser.add_argument(
        "--skip-validation-generation",
        action="store_true",
        help="Skip automatic generation on the validation split after training",
    )

    args = parser.parse_args()

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
    )

    model = FastVisionModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    dataset_name = DatasetNames(args.dataset_name)
    dataset = get_data(dataset_name, "train")
    formatted_dataset = format_dataset(dataset_name, dataset, tokenizer)

    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=args.logging_steps,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=args.seed,
        output_dir=str(args.output_dir),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can speed up training for short sequences
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    )

    trainer.train()

    model_output_dir = args.output_dir / f"{args.model_name.replace('/', '-')}_ft"
    model.save_pretrained(str(model_output_dir))
    tokenizer.save_pretrained(str(model_output_dir))

    if not args.skip_validation_generation:
        validation_output_path = (
            args.output_dir / f"{model_output_dir.name}_{args.dataset_name}_validation.jsonl"
        )
        generate_validation_outputs(
            model,
            tokenizer,
            dataset_name,
            model_output_dir,
            validation_output_path,
            max_new_tokens=args.validation_max_new_tokens,
        )
        print(f"Validation outputs saved to {validation_output_path}")


if __name__ == "__main__":
    main()
