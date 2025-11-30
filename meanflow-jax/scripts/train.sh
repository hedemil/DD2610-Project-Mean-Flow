#!/bin/bash
# Training script for MeanFlow-JAX
# Usage: bash scripts/train.sh [dataset] [experiment_name]
#
# Datasets:
#   mnist      - 2D MNIST (32x32 grayscale, 60k samples)
#   mnist3d    - 3D-MNIST (64x64 grayscale, 10k samples)
#   imagenet   - ImageNet (256x256 RGB, 1.3M samples, requires VAE)
#
# Example:
#   bash scripts/train.sh mnist my_mnist_experiment
#   bash scripts/train.sh mnist3d test_3d
#

DATASET=${1:-mnist}
EXP_NAME=${2:-${DATASET}_$(date '+%Y%m%d_%H%M%S')}

# Project root
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_ROOT"

# Set up logging directory
LOG_DIR="logs/${EXP_NAME}"
mkdir -p "${LOG_DIR}"

echo "========================================"
echo "  MeanFlow-JAX Training"
echo "========================================"
echo "Dataset:     ${DATASET}"
echo "Experiment:  ${EXP_NAME}"
echo "Log dir:     ${LOG_DIR}"
echo "========================================"
echo ""

# XLA flags for GPU memory management
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run training
python3 main.py \
    --workdir="${LOG_DIR}" \
    --config="configs/load_config.py:${DATASET}" \
    2>&1 | tee "${LOG_DIR}/output.log"

echo ""
echo "========================================"
echo "Training completed!"
echo "Logs saved to: ${LOG_DIR}"
echo "========================================"
