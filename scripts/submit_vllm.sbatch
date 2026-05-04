#!/bin/bash

# This file is designed for the ISI Slurm Compute Cluster which Jacob has access to.
# I am permitted to (reasonably) use compute cluster resources for my academic work.
# This enables me to run gemma models on GPUs with 40/48 GB of VRAM.
# Our cluster is setup with apptainer so we rebuilt the docker container into one apptainer can work with
# To build the sif file
# apptainer build vllm.sif docker://vllm/vllm-openai:latest

#SBATCH --job-name=vllm_container
#SBATCH --partition=medium-lg
#SBATCH --account=eclair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/vllm_%j.out
#SBATCH --error=slurm/vllm_%j.err

VLLM_IMAGE="vllm.sif"
# VLLM_PORT=8800

# VLLM specific arguments
# Replace with your desired model and other configurations.
# Set VLLM_MODEL to the base model name or a local merged checkpoint path.
# For LoRA adapters, also set VLLM_LORA_PATH and optionally VLLM_LORA_NAME.
# VLLM_MODEL="google/gemma-3-12b-it"
# VLLM_LORA_PATH="/abs/path/to/adapter"
# VLLM_LORA_NAME="gemma3-ft"
# VLLM_MAX_LORA_RANK=16
# VLLM_MAX_MODEL_LEN=2048
# VLLM_GPU_MEMORY_UTILIZATION=0.8
# VLLM_ENFORCE_EAGER=1
VLLM_TENSOR_PARALLEL_SIZE=1 # Adjust based on your model and GPU count

echo "Starting vLLM service..."
echo "Using Apptainer image: ${VLLM_IMAGE}"
echo "Model: ${VLLM_MODEL}"
echo "Tensor Parallel Size: ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "Listening on port: ${VLLM_PORT}"
if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  echo "Max Model Length: ${VLLM_MAX_MODEL_LEN}"
fi
if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  echo "GPU Memory Utilization: ${VLLM_GPU_MEMORY_UTILIZATION}"
fi
if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" || "${VLLM_ENFORCE_EAGER:-}" == "true" ]]; then
  echo "CUDA Graphs: disabled via --enforce-eager"
fi

APPTAINER_BIND_ARGS=(--bind "${HF_HOME}:${HF_HOME}")
VLLM_ARGS=(
  --host "$(hostname -f)"
  --port "${VLLM_PORT}"
  --model "${VLLM_MODEL}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
)

if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
  VLLM_ARGS+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  VLLM_ARGS+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

if [[ "${VLLM_ENFORCE_EAGER:-0}" == "1" || "${VLLM_ENFORCE_EAGER:-}" == "true" ]]; then
  VLLM_ARGS+=(--enforce-eager)
fi

if [[ -n "${VLLM_LORA_PATH:-}" ]]; then
  if [[ -z "${VLLM_MODEL:-}" ]]; then
    echo "VLLM_MODEL must be set to the base model when using VLLM_LORA_PATH." >&2
    exit 1
  fi

  VLLM_LORA_NAME="${VLLM_LORA_NAME:-$(basename "${VLLM_LORA_PATH}")}"
  VLLM_MAX_LORA_RANK="${VLLM_MAX_LORA_RANK:-16}"
  VLLM_LORA_PARENT="$(dirname "${VLLM_LORA_PATH}")"

  echo "LoRA Adapter: ${VLLM_LORA_NAME}"
  echo "LoRA Path: ${VLLM_LORA_PATH}"
  echo "Max LoRA Rank: ${VLLM_MAX_LORA_RANK}"

  APPTAINER_BIND_ARGS+=(--bind "${VLLM_LORA_PARENT}:${VLLM_LORA_PARENT}")
  VLLM_ARGS+=(
    --enable-lora
    --max-lora-rank "${VLLM_MAX_LORA_RANK}"
    --lora-modules "${VLLM_LORA_NAME}=${VLLM_LORA_PATH}"
  )
fi

# Run the vLLM service using Apptainer.
apptainer exec --nv \
  "${APPTAINER_BIND_ARGS[@]}" \
  "${VLLM_IMAGE}" \
  python3 -m vllm.entrypoints.openai.api_server \
  "${VLLM_ARGS[@]}"
