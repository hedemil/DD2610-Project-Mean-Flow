"""Extract training metrics from TensorBoard events file."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing import event_accumulator
import matplotlib.pyplot as plt

def extract_metrics(logdir, output_dir='training_metrics'):
    """Extract all metrics from TensorBoard events file."""

    # Find event file
    event_file = None
    for f in os.listdir(logdir):
        if f.startswith('events.out.tfevents'):
            event_file = os.path.join(logdir, f)
            break

    if not event_file:
        print(f"No event file found in {logdir}")
        return

    print(f"Reading event file: {event_file}")

    # Load events
    ea = event_accumulator.EventAccumulator(
        event_file,
        size_guidance={event_accumulator.TENSORS: 0}
    )
    ea.Reload()

    # Get available tags
    tensor_tags = ea.Tags()['tensors']
    print(f"\nAvailable metrics: {tensor_tags}")

    # Extract each metric
    metrics_data = {}
    for tag in tensor_tags:
        if tag == 'vis_sample':  # Skip image data
            continue

        events = ea.Tensors(tag)
        steps = []
        values = []

        for event in events:
            steps.append(event.step)
            tensor_proto = event.tensor_proto
            value = tf.make_ndarray(tensor_proto)
            # Handle both scalar and array values
            if value.size == 1:
                values.append(float(value.flatten()[0]))
            else:
                values.append(float(value.mean()))

        metrics_data[tag] = {'steps': steps, 'values': values}
        print(f"  {tag}: {len(values)} records")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    csv_path = os.path.join(output_dir, 'training_metrics.csv')

    # Find common steps across all metrics
    all_steps = sorted(set(metrics_data['loss']['steps']))

    # Create DataFrame
    df_dict = {'step': all_steps}
    for tag in metrics_data:
        step_to_value = dict(zip(metrics_data[tag]['steps'], metrics_data[tag]['values']))
        df_dict[tag] = [step_to_value.get(step, np.nan) for step in all_steps]

    df = pd.DataFrame(df_dict)
    df.to_csv(csv_path, index=False)
    print(f"\nSaved metrics to: {csv_path}")

    # Save individual numpy arrays
    for tag in metrics_data:
        npy_path = os.path.join(output_dir, f'{tag}.npy')
        data = np.column_stack([metrics_data[tag]['steps'], metrics_data[tag]['values']])
        np.save(npy_path, data)
        print(f"  Saved: {tag}.npy")

    # Create plots
    print("\nCreating plots...")

    # Convert lists to numpy for easier manipulation
    for tag in metrics_data:
        metrics_data[tag]['steps'] = np.array(metrics_data[tag]['steps'])
        metrics_data[tag]['values'] = np.array(metrics_data[tag]['values'])

    window = 100

    # Plot 1: Loss and V_Loss Comparison
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Loss (normalized) - actual values
    ax = axes[0]
    loss_steps = metrics_data['loss']['steps']
    loss_values = metrics_data['loss']['values']
    ax.plot(loss_steps, loss_values, alpha=0.3, linewidth=0.5, color='blue', label='Raw')
    smoothed_loss = np.convolve(loss_values, np.ones(window)/window, mode='same')
    ax.plot(loss_steps, smoothed_loss, linewidth=2, color='darkblue', label=f'Smoothed (window={window})')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # V_Loss (velocity loss)
    ax = axes[1]
    vloss_steps = metrics_data['v_loss']['steps']
    vloss_values = metrics_data['v_loss']['values']
    ax.plot(vloss_steps, vloss_values, alpha=0.3, linewidth=0.5, color='red', label='Raw')
    smoothed_vloss = np.convolve(vloss_values, np.ones(window)/window, mode='same')
    ax.plot(vloss_steps, smoothed_vloss, linewidth=2, color='darkred', label=f'Smoothed (window={window})')
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('V_Loss (Velocity MSE)', fontsize=12)
    ax.set_title('Velocity Loss Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  Created: loss_comparison.png")
    plt.close()

    # Plot 2: All metrics in grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Loss (actual values)
    ax = axes[0, 0]
    ax.plot(loss_steps, loss_values, alpha=0.5, linewidth=0.5, color='blue')
    ax.plot(loss_steps, smoothed_loss, linewidth=2, color='darkblue')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # V_Loss
    ax = axes[0, 1]
    ax.plot(vloss_steps, vloss_values, alpha=0.5, linewidth=0.5, color='red')
    ax.plot(vloss_steps, smoothed_vloss, linewidth=2, color='darkred')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('V_Loss (Velocity MSE)')
    ax.set_title('Velocity Loss', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 0]
    if 'lr' in metrics_data:
        lr_steps = metrics_data['lr']['steps']
        lr_values = metrics_data['lr']['values']
        ax.plot(lr_steps, lr_values, linewidth=2, color='green')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Steps per second
    ax = axes[1, 1]
    if 'steps_per_second' in metrics_data:
        sps_steps = metrics_data['steps_per_second']['steps']
        sps_values = metrics_data['steps_per_second']['values']
        ax.plot(sps_steps, sps_values, alpha=0.5, linewidth=0.5, color='purple')
        smoothed_sps = np.convolve(sps_values, np.ones(window)/window, mode='same')
        ax.plot(sps_steps, smoothed_sps, linewidth=2, color='darkviolet')
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Steps/Second')
        ax.set_title('Training Speed', fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics.png'), dpi=150, bbox_inches='tight')
    print(f"  Created: all_metrics.png")
    plt.close()

    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING METRICS SUMMARY")
    print("="*60)

    loss_values = metrics_data['loss']['values']
    print(f"\nLoss (Normalized):")
    print(f"  Initial: {loss_values[0]:.8f}")
    print(f"  Final: {loss_values[-1]:.8f}")
    print(f"  Min: {loss_values.min():.8f}")
    print(f"  Max: {loss_values.max():.8f}")
    print(f"  Mean: {loss_values.mean():.8f}")
    print(f"  Std: {loss_values.std():.8f}")

    vloss_values = metrics_data['v_loss']['values']
    print(f"\nV_Loss (Variance):")
    print(f"  Initial: {vloss_values[0]:.8f}")
    print(f"  Final: {vloss_values[-1]:.8f}")
    print(f"  Min: {vloss_values.min():.8f}")
    print(f"  Max: {vloss_values.max():.8f}")
    print(f"  Mean: {vloss_values.mean():.8f}")
    print(f"  Std: {vloss_values.std():.8f}")

    if 'steps_per_second' in metrics_data:
        sps_values = metrics_data['steps_per_second']['values']
        print(f"\nTraining Speed:")
        print(f"  Mean: {sps_values.mean():.2f} steps/sec")
        print(f"  Min: {sps_values.min():.2f} steps/sec")
        print(f"  Max: {sps_values.max():.2f} steps/sec")

    total_steps = len(all_steps)
    if 'ep' in metrics_data:
        total_epochs = int(max(metrics_data['ep']['values']))
        print(f"\nTraining Progress:")
        print(f"  Total steps: {total_steps}")
        print(f"  Total epochs: {total_epochs}")
        print(f"  Steps per epoch: ~{total_steps // total_epochs}")

    print("\n" + "="*60)

    return df, metrics_data

if __name__ == '__main__':
    logdir = 'workdir_3d'
    extract_metrics(logdir)
