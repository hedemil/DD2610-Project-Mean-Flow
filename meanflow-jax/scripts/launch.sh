#!/bin/bash
# Quick launch script for MeanFlow-JAX training
#
# Usage: bash scripts/launch.sh [config_name] [experiment_name]
#
# Config names:
#   mnist      - 2D MNIST (32x32 grayscale)
#   mnist3d    - 3D-MNIST (64x64 grayscale)
#   run_b4     - ImageNet with DiT-B/4 (requires VAE)
#
# Example:
#   bash scripts/launch.sh mnist
#   bash scripts/launch.sh mnist my_experiment
#   bash scripts/launch.sh mnist3d test_run
#

CONFIG=${1:-mnist}
EXP_NAME=${2:-$1}

# XLA flags to avoid OOM during convolution algorithm selection
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
# Limit GPU memory growth to avoid OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${now}_${salt}_${EXP_NAME}
export LOG_DIR=logs/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 777 -R ${LOG_DIR}

echo "=========================================="
echo "  MeanFlow-JAX Training"
echo "=========================================="
echo "Config:      ${CONFIG}"
echo "Experiment:  ${EXP_NAME}"
echo "Log dir:     ${LOG_DIR}"
echo "=========================================="
echo ""

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:${CONFIG} \
    2>&1 | tee -a $LOG_DIR/output.log
