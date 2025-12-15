"""Evaluate a trained MeanFlow checkpoint on 3D MNIST.

This script computes class-aligned metrics (Chamfer Distance, IoU, diversity)
for a trained model checkpoint.

Usage:
    # Evaluate single checkpoint
    python evaluate.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1

    # Evaluate with more samples
    python evaluate.py --config configs/train_3d_v1.yml --checkpoint workdir_3d_v1 --num_samples 500
"""

import os
import sys
import argparse
import yaml
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from flax.training import checkpoints
from ml_collections import ConfigDict

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train import create_train_state, sample_step
from utils import input_pipeline
from utils.evaluation_3d import chamfer_distance_batch, compute_voxel_iou
from meanflow import MeanFlow


def load_yaml_config(yaml_path):
    """Load YAML config and convert to ConfigDict."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    def dict_to_config(d):
        if isinstance(d, dict):
            return ConfigDict({k: dict_to_config(v) for k, v in d.items()})
        elif isinstance(d, list):
            return [dict_to_config(item) for item in d]
        else:
            return d

    return dict_to_config(config_dict)


def main():
    parser = argparse.ArgumentParser(description='Evaluate 3D MeanFlow checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint directory')
    parser.add_argument('--num_samples', type=int, default=200, help='Number of samples to evaluate')
    args = parser.parse_args()

    print("="*70)
    print("3D MEANFLOW EVALUATION")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples: {args.num_samples}")

    # Load config
    config = load_yaml_config(args.config)

    # Add missing dataset attributes
    if not hasattr(config.dataset, 'num_workers'):
        config.dataset.num_workers = 4
    if not hasattr(config.dataset, 'prefetch_factor'):
        config.dataset.prefetch_factor = 2
    if not hasattr(config.dataset, 'pin_memory'):
        config.dataset.pin_memory = True

    # Create model
    print("\nCreating model...")
    model_config = {k: v for k, v in config.model.items() if k != 'cls'}
    model = MeanFlow(
        model_str=config.model.cls,
        model_config=model_config,
        num_classes=config.dataset.num_classes,
        **dict(config.method),
    )

    # Load checkpoint
    checkpoint_dir = os.path.abspath(args.checkpoint)
    print(f"Loading checkpoint from: {checkpoint_dir}")
    rng = jax.random.PRNGKey(config.training.seed)
    state = create_train_state(
        rng=rng,
        config=config,
        model=model,
        image_size=config.dataset.image_size,
        lr_value=config.training.learning_rate
    )
    state = checkpoints.restore_checkpoint(checkpoint_dir, state)
    print(f"✓ Loaded checkpoint at step: {state.step}")

    # Load validation data
    print("\nLoading validation data...")
    local_batch_size = config.training.batch_size // jax.process_count()
    val_loader, _ = input_pipeline.create_split(
        config.dataset,
        local_batch_size,
        split='val',
    )

    real_samples_list = []
    real_labels_list = []

    for batch_idx, batch in enumerate(val_loader):
        batch = input_pipeline.prepare_batch_data(batch)
        real_batch = batch['image']
        label_batch = batch['label']
        real_batch = real_batch.reshape(-1, *real_batch.shape[2:])
        label_batch = label_batch.reshape(-1)
        real_samples_list.append(np.array(real_batch))
        real_labels_list.append(np.array(label_batch))
        if len(real_samples_list) * real_batch.shape[0] >= args.num_samples:
            break

    real_samples = np.concatenate(real_samples_list, axis=0)[:args.num_samples]
    real_labels = np.concatenate(real_labels_list, axis=0)[:args.num_samples]
    print(f"✓ Loaded {len(real_samples)} real samples")

    # Generate samples with matching labels
    print("\nGenerating samples...")
    num_devices = jax.local_device_count()
    device_batch_size = config.fid.device_batch_size

    p_sample_step = jax.pmap(
        partial(sample_step,
                model=model,
                rng_init=jax.random.PRNGKey(config.sampling.seed),
                device_batch_size=device_batch_size,
                config=config),
        axis_name='batch'
    )

    state_rep = jax.device_put_replicated({"params": state.ema_params}, jax.local_devices())

    eval_samples = []
    eval_labels = []
    samples_per_iter = num_devices * device_batch_size
    num_iters = (len(real_labels) + samples_per_iter - 1) // samples_per_iter

    for i in range(num_iters):
        start_idx = i * samples_per_iter
        end_idx = min(start_idx + samples_per_iter, len(real_labels))
        batch_labels = real_labels[start_idx:end_idx]
        actual_batch_size = len(batch_labels)

        if len(batch_labels) < samples_per_iter:
            pad_size = samples_per_iter - len(batch_labels)
            batch_labels = np.concatenate([batch_labels, batch_labels[:pad_size]])

        batch_labels_reshaped = batch_labels.reshape(num_devices, device_batch_size)
        batch_labels_jax = jnp.array(batch_labels_reshaped)
        sample_idx = jnp.arange(num_devices) + i * num_devices

        latent = p_sample_step(state_rep, sample_idx=sample_idx, class_idx=batch_labels_jax)
        latent = latent.reshape(-1, *latent.shape[2:])
        samples = latent.transpose(0, 2, 3, 1)

        samples_np = np.array(samples)[:actual_batch_size]
        eval_samples.append(samples_np)
        eval_labels.append(real_labels[start_idx:end_idx])

    eval_samples = np.concatenate(eval_samples, axis=0)
    eval_labels = np.concatenate(eval_labels, axis=0)
    eval_samples = np.clip(eval_samples, -1.0, 1.0)
    print(f"✓ Generated {len(eval_samples)} samples")

    # Compute metrics
    threshold = config.evaluation.get('chamfer_threshold', 0.0)
    num_classes = config.dataset.num_classes

    print("\n" + "="*70)
    print("COMPUTING METRICS (CLASS-ALIGNED)")
    print("="*70)

    # Chamfer Distance
    cd_mean, cd_std = chamfer_distance_batch(
        eval_samples, real_samples,
        eval_labels, real_labels,
        threshold=threshold,
        class_aligned=True
    )

    # IoU
    iou_mean, iou_std = compute_voxel_iou(
        eval_samples, real_samples,
        eval_labels, real_labels,
        threshold=threshold,
        class_aligned=True
    )

    # Coverage
    class_counts = np.bincount(eval_labels, minlength=num_classes)
    covered = np.sum(class_counts > 0)

    # Print results
    print(f"\n{'Metric':<25} {'Value':<15}")
    print("-"*40)
    print(f"{'Chamfer Distance':<25} {cd_mean:.4f} ± {cd_std:.4f}")
    print(f"{'Voxel IoU':<25} {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"{'Mode Coverage':<25} {covered}/{num_classes}")
    print(f"{'Sample Range':<25} [{eval_samples.min():.3f}, {eval_samples.max():.3f}]")

    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nInterpretation:")
    print("  • Chamfer Distance: Lower is better (typical: 0.13-0.14)")
    print("  • IoU: Higher is better (typical: 0.20-0.22)")
    print("  • Coverage: Should be 10/10 for no mode collapse")
    print("="*70)


if __name__ == "__main__":
    main()
