# from absl import logging
import os

import jax
import jax.numpy as jnp
import numpy as np

from utils.logging_util import log_for_0


def generate_fid_samples(state, workdir, config, p_sample_step, run_p_sample_step, ema=True):
  num_steps = np.ceil(config.fid.num_samples / config.fid.device_batch_size / jax.device_count()).astype(int)

  output_dir = os.path.join(workdir, 'samples')
  os.makedirs(output_dir, exist_ok=True)

  log_for_0('Note: the first sample may be significant slower')

  # Generate first batch to determine shape
  sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())
  sample_idx = jax.device_count() * 0 + sample_idx
  log_for_0(f'Sampling step 0 / {num_steps}...')
  first_samples = run_p_sample_step(p_sample_step, state, sample_idx=sample_idx, ema=ema)
  first_samples = jax.device_get(first_samples)

  # Pre-allocate memory-mapped file to avoid loading everything into RAM
  total_samples = num_steps * first_samples.shape[0]
  sample_shape = first_samples.shape[1:]  # (H, W, C)
  samples_file = os.path.join(output_dir, 'samples_all.npy')

  log_for_0(f'Creating memory-mapped array: shape=({total_samples}, {sample_shape}), dtype=uint8')
  log_for_0(f'Estimated file size: {total_samples * np.prod(sample_shape) / (1024**3):.2f} GB')

  # Create memory-mapped array (writes directly to disk, no RAM needed)
  fp = np.lib.format.open_memmap(samples_file, mode='w+', dtype=np.uint8,
                                   shape=(total_samples,) + sample_shape)

  # Write first batch
  start_idx = 0
  end_idx = first_samples.shape[0]
  fp[start_idx:end_idx] = first_samples
  log_for_0(f'Wrote batch 0: samples {start_idx} to {end_idx}')

  # Generate and write remaining batches directly to memory-mapped file
  for step in range(1, num_steps):
    sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())
    sample_idx = jax.device_count() * step + sample_idx
    log_for_0(f'Sampling step {step} / {num_steps}...')

    samples = run_p_sample_step(p_sample_step, state, sample_idx=sample_idx, ema=ema)
    samples = jax.device_get(samples)

    # Write directly to memmap (disk), no memory accumulation
    start_idx = step * first_samples.shape[0]
    end_idx = start_idx + samples.shape[0]
    fp[start_idx:end_idx] = samples

    # Log progress every 500 steps
    if (step + 1) % 500 == 0:
      log_for_0(f'Progress: {step + 1}/{num_steps} batches written to disk')
      # Explicitly flush to disk
      fp.flush()

  # Final flush
  fp.flush()
  log_for_0(f'All {total_samples} samples written to {samples_file}')

  # Load as memory-mapped array for FID computation (no RAM needed)
  samples_all = np.load(samples_file, mmap_mode='r')
  log_for_0(f'Loaded samples as memory-mapped array: {samples_all.shape}')

  return samples_all

