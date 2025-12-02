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

  # Save samples incrementally to avoid OOM
  samples_file = os.path.join(output_dir, 'samples_all.npy')
  samples_batches = []

  log_for_0('Note: the first sample may be significant slower')
  for step in range(num_steps):
    sample_idx = jax.process_index() * jax.local_device_count() + jnp.arange(jax.local_device_count())
    sample_idx = jax.device_count() * step + sample_idx
    log_for_0(f'Sampling step {step} / {num_steps}...')
    samples = run_p_sample_step(p_sample_step, state, sample_idx=sample_idx, ema=ema)
    samples = jax.device_get(samples)
    samples_batches.append(samples)

    # Save in chunks of 100 batches to avoid memory issues
    if len(samples_batches) >= 100:
      log_for_0(f'Saving batch chunk at step {step}...')
      chunk = np.concatenate(samples_batches, axis=0)
      if step < 100:
        # First chunk - create new file
        np.save(samples_file, chunk)
      else:
        # Subsequent chunks - append
        existing = np.load(samples_file)
        combined = np.concatenate([existing, chunk], axis=0)
        np.save(samples_file, combined)
      samples_batches = []

  # Save remaining samples
  if samples_batches:
    log_for_0(f'Saving final batch chunk...')
    chunk = np.concatenate(samples_batches, axis=0)
    if os.path.exists(samples_file):
      existing = np.load(samples_file)
      samples_all = np.concatenate([existing, chunk], axis=0)
    else:
      samples_all = chunk
    np.save(samples_file, samples_all)
  else:
    samples_all = np.load(samples_file)

  log_for_0(f'All samples saved to {samples_file}')
  # samples_all = samples_all[:config.fid.num_samples]
  return samples_all

