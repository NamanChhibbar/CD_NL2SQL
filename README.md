# CD NL2SQL

## Project Setup

This project currently supports Python 3.13.x.

### Using `uv` (Recommended)

If you have [`uv`](https://docs.astral.sh/uv/) installed:

1. Sync the environment and install dependencies:
   ```bash
   uv sync
   ```
2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```

### Using `conda`

1. Create a new conda environment with Python 3.13:
   ```bash
   conda create -n cd_nl2sql python=3.13
   ```
2. Activate the environment:
   ```bash
   conda activate cd_nl2sql
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Pre-commit Configuration

This project uses [`pre-commit`](https://pre-commit.com/) to maintain code quality. To set it up:

1. Install `pre-commit` (it is included in the project dependencies):
   ```bash
   pip install pre-commit
   ```
2. Install the git hooks:
   ```bash
   pre-commit install
   ```

You can also run the hooks manually on all files:

```bash
pre-commit run --all-files
```

## Hugging Face Authentication

Gemma models are gated and require a Hugging Face token for authentication.

### Obtaining a Token

1. Create a [Hugging Face account](https://huggingface.co/join) if you don't have one.
2. Visit [Gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) and accept the license terms.
3. Generate a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens).

### Configuring the Token

#### Using the CLI (Recommended)

You can log in via the Hugging Face CLI:

```bash
huggingface-cli login
```

#### Using Environment Variables

Alternatively, you can set the `HF_TOKEN` environment variable:

```bash
export HF_TOKEN="your_token_here"
```

### Using with Docker

To use your token with vLLM in Docker, pass the `HF_TOKEN` environment variable:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model google/gemma-3-12b-it
```

## Using vLLM

You can run vLLM using Docker to serve models. This is particularly useful for running large models like Gemma on GPUs.

### Running with Docker

To run the vLLM OpenAI-compatible API server:

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN=$HF_TOKEN \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model google/gemma-3-12b-it
```

### Configuration Options

- `-e HF_TOKEN=$HF_TOKEN`: Passes your Hugging Face token to the container (required for gated models).
- `--model`: The Hugging Face model ID (e.g., `google/gemma-3-12b-it`).
- `-p 8000:8000`: Maps the container's port 8000 to the host.
- `-v ~/.cache/huggingface:/root/.cache/huggingface`: Mounts your local Hugging Face cache to avoid re-downloading models.
- `--tensor-parallel-size`: If you have multiple GPUs, you can specify the number of GPUs to use (e.g., `--tensor-parallel-size 2`).

### Serving Fine-Tuned LoRA Adapters

The training script saves PEFT LoRA adapters, not merged standalone checkpoints. To serve a fine-tuned model with vLLM, launch the base model and register the adapter with a LoRA alias:

```bash
sbatch --export=ALL,\
VLLM_MODEL=unsloth/gemma-3-270m-it-unsloth-bnb-4bit,\
VLLM_LORA_PATH=/abs/path/to/outputs/google-gemma-3-270m-it/google-gemma-3-270m-it_ft,\
VLLM_LORA_NAME=gemma3-270m-ft,\
VLLM_MAX_LORA_RANK=16 \
scripts/submit_vllm.sbatch
```

Requests should then use `model="gemma3-270m-ft"`. If you serve a merged checkpoint instead, set `VLLM_MODEL` to that checkpoint path and omit the LoRA variables.

If startup fails with CUDA OOM during `capture_model` or CUDA graph warmup, retry with a smaller served context length and eager mode:

```bash
sbatch --export=ALL,\
VLLM_MODEL=unsloth/gemma-3-270m-it-unsloth-bnb-4bit,\
VLLM_LORA_PATH=/abs/path/to/outputs/google-gemma-3-270m-it/google-gemma-3-270m-it_ft,\
VLLM_LORA_NAME=gemma3-270m-ft,\
VLLM_MAX_LORA_RANK=16,\
VLLM_MAX_MODEL_LEN=2048,\
VLLM_GPU_MEMORY_UTILIZATION=0.8,\
VLLM_ENFORCE_EAGER=1 \
scripts/submit_vllm.sbatch
```

## Running the Analysis Script

The SQL analysis scripts live under `scoring_system/`. The entry point is `scoring_system/main.py`, which evaluates every `*.jsonl` file under `scoring_system/data/`.

### How to Run

Run the analysis script from inside the `scoring_system/` directory.

```bash
cd scoring_system
python main.py
```

If you are using `uv`, you can also run:

```bash
cd scoring_system
uv run python main.py
```

### What the Script Prints

For each JSONL file in `scoring_system/data/`, the script prints:

- an error breakdown reported as fractions of all examples;
- component-wise scores for `agg`, `select`, `distinct`, `where_col`, `where_op`, `where_val`, `group_by`, `order_by`, and `limit`
- a `sql_syntax_valid` score, which is the fraction of responses that `sqlglot` can parse cleanly as SQLite SQL
- a `logical_form` score, which is the strict all-or-nothing match rate

It also writes grouped bar charts for both `logical_form` and `sql_syntax_valid`, using the same per-model/per-variant layout.

## Generating Outputs

> **Prerequisites:** activate the venv (`source .venv/bin/activate`) and make sure a vLLM server is running. All scripts must be run as **modules** from the **project root** (`python -m scripts.<name>`).

### Basic Command

```bash
python -m scripts.generate_outputs \
    --model-name <MODEL> \
    --dataset-name <DATASET> \
    --dataset-split <SPLIT> \
    --endpoint <VLLM_URL> \
    --output-dir <DIR> \
    [--guided-decoding | --agent-critic]
```

### Full Flag Reference

| Flag | Required | Default | Choices / Description |
| --- | --- | --- | --- |
| `--model-name` | No | `google/gemma-3-270m-it` | `google/gemma-3-270m-it`, `google/gemma-3-1b-it`, `google/gemma-3-4b-it`, `google/gemma-3-12b-it`, `google/gemma-3-27b-it` -- must match the model served by vLLM |
| `--dataset-name` | No | `wikisql` | `wikisql` or `SQaLe` |
| `--dataset-split` | No | `validation` | `train`, `validation`, or `test` |
| `--endpoint` | **Yes** | -- | vLLM base URL (can be passed multiple times for load balancing) |
| `--output-dir` | **Yes** | -- | Directory to write results |
| `--guided-decoding` | No | off | Enable EBNF grammar-constrained decoding |
| `--agent-critic` | No | off | Enable an agent-critic loop where SQLite syntax validation feeds errors back to the model |
| `--agent-critic-rounds` | No | `3` | Maximum repair rounds when `--agent-critic` is enabled |
| `--grammar-path` | No | WikiSQL grammar for `wikisql`, general SQL grammar otherwise | Custom grammar template (only used with `--guided-decoding`) |
| `--max-completion-tokens` | No | `512` | Max tokens per response |
| `--temperature` | No | `0.0` | Sampling temperature for every run |
| `--num-jobs` | No | `12` | Number of parallel workers |
| `--max-items` | No | all rows | Limit generation to the first N examples |

### Examples

Generate outputs on WikiSQL validation with guided decoding using the 12B model:

```bash
python -m scripts.generate_outputs \
    --model-name google/gemma-3-12b-it \
    --dataset-name wikisql \
    --dataset-split validation \
    --endpoint http://127.0.0.1:8000/v1 \
    --output-dir outputs/ \
    --guided-decoding
```

Generate outputs on SQaLe test without guided decoding using the default model:

```bash
python -m scripts.generate_outputs \
    --dataset-name SQaLe \
    --dataset-split test \
    --endpoint http://127.0.0.1:8000/v1 \
    --output-dir outputs/
```

Generate outputs with the agent-critic loop using the 4B model:

```bash
python -m scripts.generate_outputs \
    --model-name google/gemma-3-4b-it \
    --dataset-name wikisql \
    --dataset-split validation \
    --endpoint http://eclairlg02.isi.edu:8802/v1 \
    --output-dir outputs/ \
    --agent-critic
```

### Output Format

Results are saved as JSONL to `<output-dir>/<model>_<dataset>_<split>[_guided|_agent_critic].jsonl`. Each line contains:

```json
{
  "prompt": "...",
  "response": "...",
  "human_sql": "...",
  "metadata": {
    "model_name": "...",
    "used_guided_decoding": false,
    "generation_approach": "agent_critic",
    "temperature": 0.0,
    "max_output_tokens": 256,
    "response_status": "...",
    "incomplete_reason": null,
    "output_tokens": 42
    "agent_critic_rounds": 2,
    "final_validation_error": null
  },
  "query_details": {
    "dataset_name": "...",
    "dataset_index": 0,
    "raw_question": "...",
    "schema_or_table_details": "..."
  }
}
```

### Smoke Test (Hardcoded Examples)

To quickly verify guided decoding against a few built-in samples without loading a full dataset:

```bash
python -m scripts.test_guided_decoding --base-url http://127.0.0.1:8000/v1
```

## Training

Run training from the project root:

```bash
python training/train.py \
    --model-name google/gemma-3-4b-it \
    --dataset-name wikisql \
    --output-dir outputs/train_run
```

After training completes, the script now also runs the fine-tuned model on the validation split automatically and writes JSONL output to `<output-dir>/<saved-model-dir>_<dataset>_validation.jsonl`.

Additional training flags:

| Flag | Required | Default | Description |
| --- | --- | --- | --- |
| `--validation-max-new-tokens` | No | `256` | Max generated tokens per validation example |
| `--skip-validation-generation` | No | off | Skip the automatic post-training validation generation step |
