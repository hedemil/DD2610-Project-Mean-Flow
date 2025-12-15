"""Evaluate all configs and create comparison table.

Usage:
    python eval_all_configs.py
"""

import os
import sys
import subprocess
import json
import yaml
import numpy as np
from pathlib import Path


CONFIGS = [
    {
        'name': 'V1 (Baseline)',
        'config': 'configs/train_3d_v1.yml',
        'checkpoint': 'workdir_3d_v1',
        'description': 'Baseline with CFG'
    },
    {
        'name': 'V3 (No CFG)',
        'config': 'configs/train_3d_v3.yml',
        'checkpoint': 'workdir_3d_v3',
        'description': 'Ablation: No classifier-free guidance'
    },
    {
        'name': 'V4',
        'config': 'configs/train_3d_v4.yml',
        'checkpoint': 'workdir_3d_v4',
        'description': 'Config V4'
    },
    {
        'name': 'V5',
        'config': 'configs/train_3d_v5.yml',
        'checkpoint': 'workdir_3d_v5',
        'description': 'Config V5'
    },
]


def check_checkpoint_exists(checkpoint_dir):
    """Check if checkpoint directory exists."""
    if not os.path.exists(checkpoint_dir):
        return False
    # Check for checkpoint files
    checkpoint_files = list(Path(checkpoint_dir).glob('checkpoint_*'))
    return len(checkpoint_files) > 0


def run_evaluation(config_path, checkpoint_dir, num_samples=200):
    """Run evaluation script and capture metrics."""
    print(f"\n{'='*70}")
    print(f"Running: {config_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"{'='*70}")

    # Run the improved evaluation script
    cmd = [
        sys.executable,
        'eval_improved.py',
        '--config', config_path,
        '--checkpoint', checkpoint_dir,
        '--num_samples', str(num_samples)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # Parse metrics from output
        metrics = {}

        # Look for Chamfer Distance
        for line in output.split('\n'):
            if 'Chamfer Distance:' in line:
                parts = line.split(':')[1].strip().split('±')
                if len(parts) == 2:
                    metrics['chamfer_mean'] = float(parts[0].strip())
                    metrics['chamfer_std'] = float(parts[1].strip())

            elif 'IoU:' in line and 'Per-class' not in line:
                parts = line.split(':')[1].strip().split('±')
                if len(parts) == 2:
                    metrics['iou_mean'] = float(parts[0].strip())
                    metrics['iou_std'] = float(parts[1].strip())

            elif 'Covered classes:' in line:
                parts = line.split(':')[1].strip().split('/')
                if len(parts) == 2:
                    metrics['covered_classes'] = int(parts[0].strip())

            elif 'Distribution entropy:' in line:
                parts = line.split(':')[1].strip().split('(')[0].strip()
                metrics['entropy'] = float(parts)

            elif 'Mean intra-class diversity:' in line:
                parts = line.split(':')[1].strip()
                metrics['diversity'] = float(parts)

            elif 'EMD:' in line:
                parts = line.split(':')[1].strip()
                try:
                    metrics['emd'] = float(parts)
                except:
                    pass

            elif 'Hausdorff:' in line and 'Distance' in line:
                parts = line.split(':')[1].strip().split('±')
                if len(parts) == 2:
                    metrics['hausdorff_mean'] = float(parts[0].strip())
                    metrics['hausdorff_std'] = float(parts[1].strip())

        print(output)
        return metrics, output

    except subprocess.TimeoutExpired:
        print("ERROR: Evaluation timed out!")
        return {}, ""
    except Exception as e:
        print(f"ERROR: {e}")
        return {}, ""


def print_comparison_table(all_results):
    """Print a nice comparison table."""
    print("\n" + "="*100)
    print("COMPARISON TABLE - ALL CONFIGS")
    print("="*100)

    # Header
    headers = ['Config', 'Chamfer↓', 'IoU↑', 'Coverage', 'Entropy↑', 'Diversity↑', 'EMD↓', 'Hausdorff↓']
    print(f"\n{'Config':<20} {'Chamfer↓':<15} {'IoU↑':<15} {'Coverage':<10} {'Entropy↑':<10} {'Diversity↑':<12} {'EMD↓':<12} {'Hausdorff↓':<15}")
    print("-"*100)

    for result in all_results:
        name = result['name']
        metrics = result['metrics']

        chamfer = f"{metrics.get('chamfer_mean', float('nan')):.4f}" if 'chamfer_mean' in metrics else "N/A"
        iou = f"{metrics.get('iou_mean', float('nan')):.4f}" if 'iou_mean' in metrics else "N/A"
        coverage = f"{metrics.get('covered_classes', 0)}/10" if 'covered_classes' in metrics else "N/A"
        entropy = f"{metrics.get('entropy', float('nan')):.4f}" if 'entropy' in metrics else "N/A"
        diversity = f"{metrics.get('diversity', float('nan')):.4f}" if 'diversity' in metrics else "N/A"
        emd = f"{metrics.get('emd', float('nan')):.6f}" if 'emd' in metrics else "N/A"
        hausdorff = f"{metrics.get('hausdorff_mean', float('nan')):.4f}" if 'hausdorff_mean' in metrics else "N/A"

        print(f"{name:<20} {chamfer:<15} {iou:<15} {coverage:<10} {entropy:<10} {diversity:<12} {emd:<12} {hausdorff:<15}")

    print("="*100)
    print("\nLEGEND:")
    print("  ↓ = Lower is better")
    print("  ↑ = Higher is better")
    print("\nKEY METRICS TO COMPARE:")
    print("  • Chamfer Distance: Reconstruction quality (should be < 1.0 for normalized)")
    print("  • IoU: Overlap with real samples (should be > 0.5)")
    print("  • Coverage: All 10 digits should be covered")
    print("  • Entropy: 1.0 = uniform distribution (no mode collapse)")
    print("  • Diversity: Variation within each class (higher = more diverse)")
    print("  • EMD: Distribution distance (alternative to Chamfer)")
    print("  • Hausdorff: Worst-case distance (should be low)")
    print("\n" + "="*100)


def main():
    print("="*100)
    print("EVALUATING ALL CONFIGS")
    print("="*100)

    all_results = []

    for config_info in CONFIGS:
        name = config_info['name']
        config_path = config_info['config']
        checkpoint_dir = config_info['checkpoint']
        description = config_info['description']

        # Check if checkpoint exists
        if not check_checkpoint_exists(checkpoint_dir):
            print(f"\n⚠️  Skipping {name}: Checkpoint not found at {checkpoint_dir}")
            continue

        # Check if config exists
        if not os.path.exists(config_path):
            print(f"\n⚠️  Skipping {name}: Config not found at {config_path}")
            continue

        # Run evaluation
        metrics, output = run_evaluation(config_path, checkpoint_dir, num_samples=200)

        all_results.append({
            'name': name,
            'config': config_path,
            'checkpoint': checkpoint_dir,
            'description': description,
            'metrics': metrics,
            'output': output
        })

    # Print comparison table
    if all_results:
        print_comparison_table(all_results)

        # Save results to JSON
        results_file = 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to: {results_file}")

        # Find best configs
        print("\n" + "="*100)
        print("BEST CONFIGS:")
        print("="*100)

        # Best by Chamfer (lower is better)
        valid_chamfer = [(r['name'], r['metrics'].get('chamfer_mean', float('inf')))
                         for r in all_results if 'chamfer_mean' in r['metrics']]
        if valid_chamfer:
            best_chamfer = min(valid_chamfer, key=lambda x: x[1])
            print(f"  Best Chamfer Distance: {best_chamfer[0]} ({best_chamfer[1]:.4f})")

        # Best by IoU (higher is better)
        valid_iou = [(r['name'], r['metrics'].get('iou_mean', 0))
                     for r in all_results if 'iou_mean' in r['metrics']]
        if valid_iou:
            best_iou = max(valid_iou, key=lambda x: x[1])
            print(f"  Best IoU: {best_iou[0]} ({best_iou[1]:.4f})")

        # Best by diversity (higher is better)
        valid_div = [(r['name'], r['metrics'].get('diversity', 0))
                     for r in all_results if 'diversity' in r['metrics']]
        if valid_div:
            best_div = max(valid_div, key=lambda x: x[1])
            print(f"  Best Diversity: {best_div[0]} ({best_div[1]:.4f})")

        print("="*100)

    else:
        print("\n❌ No configs were evaluated!")


if __name__ == "__main__":
    main()
