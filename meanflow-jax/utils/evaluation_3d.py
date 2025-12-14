"""3D evaluation metrics for voxel-based generative models.

Implements:
1. Chamfer Distance: Point cloud similarity metric
2. 3D FID: Fréchet Inception Distance using 2D projections
3. IoU: Intersection over Union for voxel grids
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
import os


def voxels_to_point_cloud(voxels: jnp.ndarray, threshold: float = 0.0) -> jnp.ndarray:
    """Convert voxel grid to point cloud.

    Args:
        voxels: Voxel grid of shape (H, W, D) or (B, H, W, D)
        threshold: Voxel values above this are considered "on"

    Returns:
        Point cloud coordinates of shape (N, 3) or (B, N, 3)
    """
    if voxels.ndim == 3:
        # Single voxel grid
        coords = jnp.argwhere(voxels > threshold).astype(jnp.float32)
        return coords
    elif voxels.ndim == 4:
        # Batch of voxel grids
        batch_size = voxels.shape[0]
        point_clouds = []
        for i in range(batch_size):
            coords = jnp.argwhere(voxels[i] > threshold).astype(jnp.float32)
            point_clouds.append(coords)
        return point_clouds
    else:
        raise ValueError(f"Expected 3D or 4D voxel grid, got shape {voxels.shape}")


def chamfer_distance_single(pc1: jnp.ndarray, pc2: jnp.ndarray, normalize: bool = True) -> float:
    """Compute Chamfer Distance between two point clouds.

    Chamfer Distance = (mean(min_dist(pc1→pc2)) + mean(min_dist(pc2→pc1))) / 2

    Args:
        pc1: Point cloud 1 of shape (N1, 3)
        pc2: Point cloud 2 of shape (N2, 3)
        normalize: If True, normalize coordinates to [0, 1]

    Returns:
        Chamfer distance (scalar)
    """
    if normalize:
        # Normalize coordinates to [0, 1] range for each point cloud independently
        pc1_min, pc1_max = pc1.min(axis=0), pc1.max(axis=0)
        pc2_min, pc2_max = pc2.min(axis=0), pc2.max(axis=0)
        
        # Avoid division by zero
        pc1_range = pc1_max - pc1_min
        pc2_range = pc2_max - pc2_min
        pc1_range = jnp.where(pc1_range == 0, 1, pc1_range)
        pc2_range = jnp.where(pc2_range == 0, 1, pc2_range)
        
        pc1 = (pc1 - pc1_min) / pc1_range
        pc2 = (pc2 - pc2_min) / pc2_range
    
    # Compute pairwise squared distances: (N1, N2)
    dists_sq = jnp.sum((pc1[:, None, :] - pc2[None, :, :]) ** 2, axis=2)
    
    # For each point in pc1, find nearest point in pc2
    min_dists_sq_1_to_2 = jnp.min(dists_sq, axis=1)
    
    # For each point in pc2, find nearest point in pc1
    min_dists_sq_2_to_1 = jnp.min(dists_sq, axis=0)
    
    # Chamfer distance: average of both directions, then take sqrt
    chamfer = (jnp.mean(min_dists_sq_1_to_2) + jnp.mean(min_dists_sq_2_to_1)) / 2.0
    chamfer = jnp.sqrt(chamfer)
    
    return float(chamfer)


def chamfer_distance_batch(generated_voxels: np.ndarray,
                           real_voxels: np.ndarray,
                           generated_labels: Optional[np.ndarray] = None,
                           real_labels: Optional[np.ndarray] = None,
                           threshold: float = 0.0,
                           class_aligned: bool = True) -> Tuple[float, float]:
    """Compute Chamfer Distance between batches of generated and real voxel grids.

    Args:
        generated_voxels: Generated voxels of shape (B, H, W, D)
        real_voxels: Real voxels of shape (B, H, W, D)
        generated_labels: Optional labels for generated samples (B,)
        real_labels: Optional labels for real samples (B,)
        threshold: Voxel threshold for point cloud conversion
        class_aligned: If True and labels provided, compare only within same class

    Returns:
        (mean_chamfer, std_chamfer): Mean and std of Chamfer distances
    """
    chamfer_distances = []

    # Class-aligned comparison (RECOMMENDED)
    if class_aligned and generated_labels is not None and real_labels is not None:
        num_classes = max(generated_labels.max(), real_labels.max()) + 1

        for cls in range(num_classes):
            gen_mask = generated_labels == cls
            real_mask = real_labels == cls

            gen_cls = generated_voxels[gen_mask]
            real_cls = real_voxels[real_mask]

            if len(gen_cls) == 0 or len(real_cls) == 0:
                continue

            # Compare pairs within class
            num_pairs = min(len(gen_cls), len(real_cls))
            for i in range(num_pairs):
                gen_pc = voxels_to_point_cloud(gen_cls[i], threshold)
                real_pc = voxels_to_point_cloud(real_cls[i], threshold)

                if len(gen_pc) > 0 and len(real_pc) > 0:
                    cd = chamfer_distance_single(gen_pc, real_pc, normalize=True)
                    chamfer_distances.append(cd)

    # Simple batch comparison (legacy)
    else:
        assert generated_voxels.shape == real_voxels.shape, \
            f"Shape mismatch: {generated_voxels.shape} vs {real_voxels.shape}"

        batch_size = generated_voxels.shape[0]
        for i in range(batch_size):
            gen_pc = voxels_to_point_cloud(generated_voxels[i], threshold)
            real_pc = voxels_to_point_cloud(real_voxels[i], threshold)

            if len(gen_pc) == 0 or len(real_pc) == 0:
                continue

            cd = chamfer_distance_single(gen_pc, real_pc, normalize=True)
            chamfer_distances.append(cd)

    if len(chamfer_distances) == 0:
        return float('nan'), float('nan')

    chamfer_distances = np.array(chamfer_distances)
    return float(chamfer_distances.mean()), float(chamfer_distances.std())


def compute_voxel_iou_single(voxels1: np.ndarray,
                             voxels2: np.ndarray,
                             threshold: float = 0.0) -> float:
    """Compute Intersection over Union between two single voxel grids.

    Args:
        voxels1: First voxel grid of shape (H, W, D)
        voxels2: Second voxel grid of shape (H, W, D)
        threshold: Voxel values above this are considered "on"

    Returns:
        IoU score (0-1)
    """
    binary1 = (voxels1 > threshold).astype(np.float32)
    binary2 = (voxels2 > threshold).astype(np.float32)

    intersection = np.sum(binary1 * binary2)
    union = np.sum(np.maximum(binary1, binary2))

    if union == 0:
        return 0.0

    return float(intersection / union)


def compute_voxel_iou(voxels1: np.ndarray,
                      voxels2: np.ndarray,
                      labels1: Optional[np.ndarray] = None,
                      labels2: Optional[np.ndarray] = None,
                      threshold: float = 0.0,
                      class_aligned: bool = True) -> tuple:
    """Compute Intersection over Union between batches of voxel grids.

    Args:
        voxels1: First batch of voxel grids of shape (B, H, W, D)
        voxels2: Second batch of voxel grids of shape (B, H, W, D)
        labels1: Optional labels for first batch (B,)
        labels2: Optional labels for second batch (B,)
        threshold: Voxel values above this are considered "on"
        class_aligned: If True and labels provided, compare only within same class

    Returns:
        Tuple of (mean_iou, std_iou)
    """
    iou_scores = []

    # Class-aligned comparison (RECOMMENDED)
    if class_aligned and labels1 is not None and labels2 is not None:
        num_classes = max(labels1.max(), labels2.max()) + 1

        for cls in range(num_classes):
            mask1 = labels1 == cls
            mask2 = labels2 == cls

            cls_voxels1 = voxels1[mask1]
            cls_voxels2 = voxels2[mask2]

            if len(cls_voxels1) == 0 or len(cls_voxels2) == 0:
                continue

            # Compare pairs within class
            num_pairs = min(len(cls_voxels1), len(cls_voxels2))
            for i in range(num_pairs):
                iou = compute_voxel_iou_single(cls_voxels1[i], cls_voxels2[i], threshold)
                iou_scores.append(iou)

    # Simple batch comparison (legacy)
    else:
        assert voxels1.shape == voxels2.shape, \
            f"Shape mismatch: {voxels1.shape} vs {voxels2.shape}"

        batch_size = voxels1.shape[0]
        for i in range(batch_size):
            iou = compute_voxel_iou_single(voxels1[i], voxels2[i], threshold)
            iou_scores.append(iou)

    if len(iou_scores) == 0:
        return float('nan'), float('nan')

    iou_scores = np.array(iou_scores)
    return float(iou_scores.mean()), float(iou_scores.std())


def render_voxel_projection(voxels: np.ndarray,
                            view: str = 'xy') -> np.ndarray:
    """Render a 2D projection of a voxel grid.

    Args:
        voxels: Voxel grid of shape (H, W, D)
        view: Projection view - 'xy' (top), 'xz' (front), 'yz' (side)

    Returns:
        2D projection of shape (H, W) or (H, D) or (W, D)
    """
    if view == 'xy':
        # Top view: max projection along z-axis
        projection = voxels.max(axis=2)
    elif view == 'xz':
        # Front view: max projection along y-axis
        projection = voxels.max(axis=1)
    elif view == 'yz':
        # Side view: max projection along x-axis
        projection = voxels.max(axis=0)
    else:
        raise ValueError(f"Unknown view: {view}. Use 'xy', 'xz', or 'yz'")

    return projection


def render_multi_view(voxels: np.ndarray,
                     views: list = ['xy', 'xz', 'yz']) -> np.ndarray:
    """Render multiple 2D projections of a voxel grid.

    Args:
        voxels: Voxel grid of shape (H, W, D)
        views: List of projection views

    Returns:
        Stacked projections of shape (num_views, H_max, W_max)
        Each projection is resized to the same dimensions
    """
    projections = []
    max_h, max_w = 0, 0

    # First pass: get max dimensions
    for view in views:
        proj = render_voxel_projection(voxels, view)
        max_h = max(max_h, proj.shape[0])
        max_w = max(max_w, proj.shape[1])

    # Second pass: resize and stack
    for view in views:
        proj = render_voxel_projection(voxels, view)
        # Pad to max dimensions
        h_pad = max_h - proj.shape[0]
        w_pad = max_w - proj.shape[1]
        proj_padded = np.pad(proj, ((0, h_pad), (0, w_pad)), mode='constant')
        projections.append(proj_padded)

    return np.stack(projections)


def prepare_fid_images(voxels: np.ndarray,
                      views: list = ['xy', 'xz', 'yz']) -> np.ndarray:
    """Prepare voxel grids as 2D images for FID calculation.

    Args:
        voxels: Voxel grids of shape (B, H, W, D)
        views: List of projection views to use

    Returns:
        Images of shape (B * num_views, H, W, 3) in RGB format [0, 255]
    """
    batch_size = voxels.shape[0]
    all_images = []

    for i in range(batch_size):
        # Render multi-view projections
        projections = render_multi_view(voxels[i], views)  # (num_views, H, W)

        # Convert each projection to RGB image
        for proj in projections:
            # Normalize to [0, 255]
            proj_norm = ((proj - proj.min()) / (proj.max() - proj.min() + 1e-8) * 255)
            proj_norm = proj_norm.astype(np.uint8)

            # Convert grayscale to RGB by repeating channels
            proj_rgb = np.stack([proj_norm, proj_norm, proj_norm], axis=-1)
            all_images.append(proj_rgb)

    return np.stack(all_images)


def calculate_fid_3d(generated_voxels: np.ndarray,
                    real_voxels: np.ndarray,
                    inception_net,
                    batch_size: int = 50) -> float:
    """Calculate FID score between generated and real 3D voxels using 2D projections.

    Args:
        generated_voxels: Generated voxels of shape (N, H, W, D)
        real_voxels: Real voxels of shape (N, H, W, D)
        inception_net: Pre-loaded Inception network for feature extraction
        batch_size: Batch size for Inception network

    Returns:
        FID score (lower is better)
    """
    # Import FID utilities
    from utils import fid_util

    # Prepare 2D projections for both generated and real voxels
    print("  Rendering 2D projections of generated samples...")
    gen_images = prepare_fid_images(generated_voxels, views=['xy', 'xz', 'yz'])

    print("  Rendering 2D projections of real samples...")
    real_images = prepare_fid_images(real_voxels, views=['xy', 'xz', 'yz'])

    print(f"  Generated images: {gen_images.shape}")
    print(f"  Real images: {real_images.shape}")

    # Compute statistics for both distributions
    print("  Computing Inception features for generated samples...")
    mu_gen, sigma_gen = fid_util.compute_stats(gen_images, inception_net, batch_size)

    print("  Computing Inception features for real samples...")
    mu_real, sigma_real = fid_util.compute_stats(real_images, inception_net, batch_size)

    # Compute FID
    fid_score = fid_util.compute_fid(mu_gen, mu_real, sigma_gen, sigma_real)

    return float(fid_score)


# Example usage and testing
if __name__ == '__main__':
    print("Testing 3D evaluation metrics...")

    # Create dummy data
    dummy_voxels1 = np.random.randn(2, 16, 16, 16).astype(np.float32)
    dummy_voxels2 = np.random.randn(2, 16, 16, 16).astype(np.float32)

    # Test Chamfer Distance
    print("\n1. Testing Chamfer Distance...")
    cd_mean, cd_std = chamfer_distance_batch(dummy_voxels1, dummy_voxels2, threshold=-0.5)
    print(f"   Chamfer Distance: {cd_mean:.4f} ± {cd_std:.4f}")

    # Test IoU
    print("\n2. Testing IoU...")
    iou = compute_voxel_iou(dummy_voxels1[0], dummy_voxels2[0], threshold=0.0)
    print(f"   IoU: {iou:.4f}")

    # Test multi-view rendering
    print("\n3. Testing multi-view rendering...")
    fid_images = prepare_fid_images(dummy_voxels1[:1], views=['xy', 'xz', 'yz'])
    print(f"   FID images shape: {fid_images.shape}")

    print("\n✓ All tests passed!")
