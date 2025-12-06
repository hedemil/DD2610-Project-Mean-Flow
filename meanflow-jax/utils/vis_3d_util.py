"""3D visualization utilities for voxel data."""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def visualize_voxel_3d(voxel_data, threshold=0.5, save_path=None, title='3D Voxel'):
    """
    Visualize voxel data as a 3D plot.

    Args:
        voxel_data: (D, H, W) or (H, W, D) numpy array with values in [0, 1] or [-1, 1]
        threshold: Voxel occupancy threshold for binary visualization
        save_path: Optional path to save figure
        title: Title for the plot
    """
    # Normalize to [0, 1] if needed
    if voxel_data.min() < 0:
        voxel_data = (voxel_data + 1) / 2  # [-1,1] -> [0,1]

    # Create binary occupancy grid
    occupied = voxel_data > threshold

    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot voxels
    ax.voxels(occupied, facecolors='blue', edgecolors='gray', alpha=0.7, linewidth=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_voxel_slices(voxel_data, save_path=None, title='Voxel Slices'):
    """
    Visualize all depth slices of voxel data in a grid.

    Args:
        voxel_data: (16, 16, 16) numpy array
        save_path: Optional path to save figure
        title: Title for the plot
    """
    # Normalize to [0, 1] if needed
    if voxel_data.min() < 0:
        voxel_data = (voxel_data + 1) / 2  # [-1,1] -> [0,1]

    depth = voxel_data.shape[2]

    # Create grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < depth:
            ax.imshow(voxel_data[:, :, i], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Slice {i}')
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_voxel_grid(voxels_list, titles=None, save_path=None, ncols=4):
    """
    Visualize multiple voxel samples in a grid (using slice view).

    Args:
        voxels_list: List of (16, 16, 16) numpy arrays
        titles: Optional list of titles
        save_path: Optional path to save figure
        ncols: Number of columns in grid
    """
    n_samples = len(voxels_list)
    nrows = (n_samples + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    for idx, voxel in enumerate(voxels_list):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        # Normalize
        if voxel.min() < 0:
            voxel = (voxel + 1) / 2

        # Show middle slice
        middle_slice = voxel.shape[2] // 2
        ax.imshow(voxel[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)

        if titles and idx < len(titles):
            ax.set_title(titles[idx])
        ax.axis('off')

    # Hide empty subplots
    for idx in range(n_samples, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def make_3d_grid_visualization(samples, grid=4):
    """
    Create a grid visualization of 3D voxel samples (middle slice view).
    Compatible with existing vis_util interface.

    Args:
        samples: (N, H, W, D) numpy array where D is depth (treated as channels)
        grid: Grid size (will show grid x grid samples)

    Returns:
        vis_image: (H_grid, W_grid, 3) RGB image for tensorboard
    """
    n_samples = min(grid * grid, len(samples))

    # Create figure
    fig, axes = plt.subplots(grid, grid, figsize=(grid * 2, grid * 2))

    for idx in range(grid * grid):
        row = idx // grid
        col = idx % grid
        ax = axes[row, col] if grid > 1 else axes

        if idx < n_samples:
            voxel = samples[idx]  # (H, W, D)

            # Normalize to [0, 1]
            if voxel.min() < 0:
                voxel = (voxel + 1) / 2

            # Show middle depth slice
            middle_slice = voxel.shape[2] // 2
            ax.imshow(voxel[:, :, middle_slice], cmap='gray', vmin=0, vmax=1)

        ax.axis('off')

    plt.tight_layout(pad=0.1)

    # Convert to numpy array
    fig.canvas.draw()
    vis_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_image = vis_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return vis_image


def compute_voxel_iou(real_voxels, fake_voxels, threshold=0.5):
    """
    Compute Intersection over Union for voxel data.

    Args:
        real_voxels: (N, H, W, D) numpy array
        fake_voxels: (N, H, W, D) numpy array
        threshold: Binarization threshold

    Returns:
        mean_iou: Average IoU across all samples
    """
    # Normalize to [0, 1]
    if real_voxels.min() < 0:
        real_voxels = (real_voxels + 1) / 2
    if fake_voxels.min() < 0:
        fake_voxels = (fake_voxels + 1) / 2

    # Binarize
    real_binary = (real_voxels > threshold).astype(float)
    fake_binary = (fake_voxels > threshold).astype(float)

    # Compute IoU for each sample
    intersection = np.sum(real_binary * fake_binary, axis=(1, 2, 3))
    union = np.sum((real_binary + fake_binary) > 0, axis=(1, 2, 3))

    iou = intersection / (union + 1e-8)

    return np.mean(iou)


if __name__ == '__main__':
    # Test with random data
    print("Testing 3D visualization utilities...")

    # Create sample voxel data
    test_voxel = np.random.rand(16, 16, 16)
    test_voxel = (test_voxel > 0.7).astype(float)  # Binary voxels

    # Test slice visualization
    visualize_voxel_slices(test_voxel, save_path='test_slices.png', title='Test Voxel Slices')
    print("✓ Saved test_slices.png")

    # Test 3D visualization
    visualize_voxel_3d(test_voxel, save_path='test_3d.png', title='Test 3D Voxel')
    print("✓ Saved test_3d.png")

    # Test grid visualization
    test_samples = [np.random.rand(16, 16, 16) for _ in range(8)]
    vis_image = make_3d_grid_visualization(np.array(test_samples), grid=4)
    plt.imsave('test_grid.png', vis_image)
    print("✓ Saved test_grid.png")

    print("\nAll tests passed!")
