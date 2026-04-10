# CD NL2SQL

## Project Setup

This project uses Python 3.14 or later.

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

1. Create a new conda environment with Python 3.14:
   ```bash
   conda create -n cd_nl2sql python=3.14
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

## Generating Outputs

All scripts are run as modules from the project root (with the venv active).

### Without Guided Decoding

```bash
python -m scripts.generate_outputs \
    --dataset-name wikisql \
    --dataset-split validation \
    --endpoint http://127.0.0.1:8000/v1 \
    --output-dir outputs/
```

### With Guided Decoding

Add `--guided-decoding` to constrain model output to valid SQL via an EBNF grammar:

```bash
python -m scripts.generate_outputs \
    --dataset-name wikisql \
    --dataset-split validation \
    --endpoint http://127.0.0.1:8000/v1 \
    --output-dir outputs/ \
    --guided-decoding
```

This works for both `wikisql` and `SQaLe` datasets. Key optional flags:

| Flag | Default | Description |
| --- | --- | --- |
| `--guided-decoding` | off | Enable grammar-constrained decoding |
| `--grammar-path` | `guided_decoding/sql_grammar.txt` | Custom EBNF grammar template |
| `--max-completion-tokens` | 256 | Max tokens per completion |
| `--num-jobs` | 12 | Parallel workers |

Output is saved as JSONL to `<output-dir>/<model>_<dataset>_<split>[_guided].jsonl`.

### Smoke Test (Hardcoded Examples)

To quickly verify guided decoding against a few samples without loading a full dataset:

```bash
python -m scripts.test_guided_decoding --base-url http://127.0.0.1:8000/v1
```
