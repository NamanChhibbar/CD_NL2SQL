#!/bin/bash

# This script is designed for the ISI Slurm Compute Cluster.
# It runs the fine-tuning script training/train.py using Unsloth and LoRA.

#SBATCH --job-name=train_gemma
#SBATCH --partition=medium-lg
#SBATCH --account=eclair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --time=24:00:00
#SBATCH --output=slurm/train_%j.out
#SBATCH --error=slurm/train_%j.err

# Training arguments
MODEL_NAME="google/gemma-3-4b-it"
DATASET_NAME="wikisql"
OUTPUT_DIR="outputs/$(date +'%Y%m%d_%H%M%S')"
BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=4
NUM_TRAIN_EPOCHS=1
LEARNING_RATE=2e-4
MAX_SEQ_LENGTH=2048

echo "Starting training job..."
echo "Model: ${MODEL_NAME}"
echo "Dataset: ${DATASET_NAME}"
echo "Output Directory: ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

source .venv/bin/activate
python3 training/train.py \
  --model-name "${MODEL_NAME}" \
  --dataset-name "${DATASET_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-size ${BATCH_SIZE} \
  --gradient-accumulation-steps ${GRADIENT_ACCUMULATION_STEPS} \
  --num-train-epochs ${NUM_TRAIN_EPOCHS} \
  --learning-rate ${LEARNING_RATE} \
  --max-seq-length ${MAX_SEQ_LENGTH}
