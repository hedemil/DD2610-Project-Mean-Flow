"""Generate 3D samples from trained checkpoint."""

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from functools import partial
from flax import jax_utils

from meanflow import MeanFlow, generate
from utils.ckpt_util import restore_checkpoint
from train import TrainState, create_train_state
import sys

def load_config(config_path):
    """Load config from YAML file."""
    import yaml
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return ml_collections.ConfigDict(config_dict)

def sample_step(variable, model, rng_init, device_batch_size, config, sample_idx):
    """Sample from the model."""
    rng_sample = jax.random.fold_in(rng_init, sample_idx)
    images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config)
    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images

def main():
    # Configuration
    config_path = 'configs/train_3d.yml'
    checkpoint_dir = 'workdir_3d'  # Will use latest checkpoint
    output_dir = 'workdir_3d/generated_samples'
    num_samples = 100  # Generate 100 samples
    device_batch_size = 2  # Samples per device

    print("="*60)
    print("3D MNIST Sample Generation")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Num samples: {num_samples}")
    print("="*60)

    # Load config
    config = load_config(config_path)

    # Create model
    model_config = config.model.to_dict()
    model_str = model_config.pop('cls')
    print(f"\nModel: {model_str}")
    print(f"Config: {model_config}")

    model = MeanFlow(
        model_str=model_str,
        model_config=model_config,
        **config.sampling,
        **config.method,
    )

    # Create dummy state and restore from checkpoint
    print(f"\nRestoring from checkpoint...")
    rng = jax.random.key(config.training.seed)
    image_size = config.dataset.image_size
    base_lr = config.training.learning_rate

    state = create_train_state(rng, config, model, image_size, lr_value=base_lr)
    state = restore_checkpoint(state, checkpoint_dir)

    step = int(state.step)
    print(f"Loaded checkpoint at step: {step}")

    # Replicate for multi-device
    state = jax_utils.replicate(state)

    # Setup sampling function
    p_sample_step = jax.pmap(
        partial(
            sample_step,
            model=model,
            rng_init=jax.random.PRNGKey(42),
            device_batch_size=device_batch_size,
            config=config,
        ),
        axis_name='batch'
    )

    # Generate samples
    print(f"\nGenerating {num_samples} samples...")
    num_devices = jax.local_device_count()
    samples_per_iter = num_devices * device_batch_size
    num_iters = (num_samples + samples_per_iter - 1) // samples_per_iter

    all_samples = []
    for i in range(num_iters):
        print(f"  Batch {i+1}/{num_iters}...")
        sample_idx = jnp.arange(num_devices) + i * num_devices

        # Sample with EMA parameters
        variable = {"params": state.ema_params}
        latent = p_sample_step(variable, sample_idx=sample_idx)
        latent = latent.reshape(-1, *latent.shape[2:])  # Flatten device dimension

        # Transpose to (B, H, W, C) and convert to numpy
        samples = latent.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        samples = np.array(samples)

        all_samples.append(samples)

    # Concatenate and trim to exact number
    all_samples = np.concatenate(all_samples, axis=0)[:num_samples]

    print(f"\nGenerated samples shape: {all_samples.shape}")
    print(f"Value range: [{all_samples.min():.3f}, {all_samples.max():.3f}]")

    # Save samples
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/samples_{step}.npy'
    np.save(output_path, all_samples)
    print(f"\nSaved samples to: {output_path}")

    # Save a few individual samples for visualization
    for i in range(min(10, num_samples)):
        individual_path = f'{output_dir}/sample_{i:03d}.npy'
        np.save(individual_path, all_samples[i])
    print(f"Saved individual samples: sample_000.npy through sample_009.npy")

    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"\nTo visualize, run:")
    print(f"  python visualize_samples.py --samples {output_path}")

if __name__ == '__main__':
    main()
