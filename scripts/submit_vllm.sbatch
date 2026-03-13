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
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=vllm_%j.out
#SBATCH --error=vllm_%j.err

VLLM_IMAGE="vllm.sif"
VLLM_PORT=8800

# VLLM specific arguments
# Replace with your desired model and other configurations
# VLLM_MODEL="google/gemma-3-12b-it"
VLLM_TENSOR_PARALLEL_SIZE=1 # Adjust based on your model and GPU count

echo "Starting vLLM service..."
echo "Using Apptainer image: ${VLLM_IMAGE}"
echo "Model: ${VLLM_MODEL}"
echo "Tensor Parallel Size: ${VLLM_TENSOR_PARALLEL_SIZE}"
echo "Listening on port: ${VLLM_PORT}"

# Run the vLLM service using Apptainer
# The --nv option enables NVIDIA GPU support within the container.
# The -p argument maps the container port to the host port.
# The --bind option can be used to mount host directories into the container
apptainer exec --nv \
  --bind "${HF_HOME}:${HF_HOME}" \
  ${VLLM_IMAGE} \
  python3 -m vllm.entrypoints.openai.api_server \
  --host $(hostname -f) \
  --port ${VLLM_PORT} \
  --model ${VLLM_MODEL} \
  --tensor-parallel-size ${VLLM_TENSOR_PARALLEL_SIZE}
