#!/usr/bin/env python3
"""Quick diagnostic script to check JAX GPU status."""

import jax
import jax.numpy as jnp

print("=" * 60)
print("JAX GPU Diagnostic")
print("=" * 60)

# Check JAX version
print(f"\nJAX version: {jax.__version__}")

# Check available backends
print(f"\nDefault backend: {jax.default_backend()}")

# Check devices
print(f"\nAll devices: {jax.devices()}")
print(f"Local devices: {jax.local_devices()}")
print(f"Device count: {jax.device_count()}")

# Check for GPU specifically
try:
    gpu_devices = jax.devices('gpu')
    print(f"\nGPU devices found: {gpu_devices}")
except:
    print("\nNo GPU devices found!")

# Try a simple computation
print("\n" + "=" * 60)
print("Testing computation...")
print("=" * 60)

x = jnp.ones((1000, 1000))
result = jnp.dot(x, x)

print(f"\nComputation device: {result.device()}")
print(f"Result shape: {result.shape}")

# Check CUDA availability
print("\n" + "=" * 60)
print("Checking CUDA libraries...")
print("=" * 60)

try:
    from jax.lib import xla_bridge
    print(f"XLA backend: {xla_bridge.get_backend().platform}")
except Exception as e:
    print(f"Error checking XLA backend: {e}")

print("\n" + "=" * 60)
