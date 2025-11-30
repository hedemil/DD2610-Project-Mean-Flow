#!/bin/bash
# CPU-only launch script for MeanFlow-JAX training
#
# Usage: bash scripts/launch_cpu.sh [config_name] [experiment_name]
#
# Config names:
#   mnist      - 2D MNIST (32x32 grayscale)
#   mnist3d    - 3D-MNIST (64x64 grayscale)
#   run_b4     - ImageNet with DiT-B/4 (requires VAE)
#
# Example:
#   bash scripts/launch_cpu.sh mnist
#   bash scripts/launch_cpu.sh mnist my_experiment
#

CONFIG=${1:-mnist}
EXP_NAME=${2:-$1}

# Force JAX to use CPU only
export JAX_PLATFORMS=cpu

# XLA flags
export XLA_FLAGS="--xla_force_host_platform_device_count=8"  # Simulate 8 CPU devices for parallelism

export now=`date '+%Y%m%d_%H%M%S'`
export salt=`head /dev/urandom | tr -dc a-z0-9 | head -c6`
export JOBNAME=${now}_${salt}_${EXP_NAME}_cpu
export LOG_DIR=logs/$USER/$JOBNAME

mkdir -p ${LOG_DIR}
chmod 777 -R ${LOG_DIR}

echo "=========================================="
echo "  MeanFlow-JAX Training (CPU ONLY)"
echo "=========================================="
echo "Config:      ${CONFIG}"
echo "Experiment:  ${EXP_NAME}"
echo "Log dir:     ${LOG_DIR}"
echo "Device:      CPU"
echo "=========================================="
echo ""

python3 main.py \
    --workdir=${LOG_DIR} \
    --config=configs/load_config.py:${CONFIG} \
    2>&1 | tee -a $LOG_DIR/output.log
