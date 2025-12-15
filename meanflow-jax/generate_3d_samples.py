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

def sample_step(variable, model, rng_init, device_batch_size, config, sample_idx, class_idx=None):
    """Sample from the model.

    Args:
        class_idx: Optional class index to generate. If None, uses random classes.
    """
    rng_sample = jax.random.fold_in(rng_init, sample_idx)
    # CRITICAL FIX: Pass class_idx to control which class to generate
    images = generate(variable, model, rng_sample, n_sample=device_batch_size, config=config, class_idx=class_idx)
    images = images.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
    return images

def voxels_to_points_with_intensity(voxel, threshold=-0.5):
    """Convert voxel grid to point cloud with intensity values.

    Only includes voxels above threshold to avoid visualizing dense background.
    For 3D MNIST: background is -1.0, foreground is > -0.5
    """
    # voxel: (H, W, D) or (16, 16, 16)
    # Returns: (N, 3) coordinates and (N,) intensity values

    # Get foreground voxels only (above threshold)
    mask = voxel > threshold
    positions = np.argwhere(mask).astype(np.float32)
    intensities = voxel[mask].astype(np.float32)

    return positions, intensities

def create_html_visualization(points, intensities, filename, title="3D Voxel Visualization"):
    """Create interactive Three.js visualization with intensity-based coloring."""

    if len(points) == 0:
        print(f"  Warning: No points to visualize for {filename}")
        return

    # Normalize intensities to [0, 1] for coloring
    int_min, int_max = intensities.min(), intensities.max()
    intensities_norm = (intensities - int_min) / (int_max - int_min + 1e-8)

    positions = points.flatten().tolist()
    colors = []
    for intensity in intensities_norm:
        # Map intensity to color: blue (low) -> cyan -> green -> yellow -> red (high)
        r = min(1.0, max(0.0, 2 * intensity - 0.5))
        g = min(1.0, max(0.0, 2 * intensity if intensity < 0.5 else 2 * (1 - intensity)))
        b = min(1.0, max(0.0, 2 * (0.5 - intensity)))
        colors.extend([r, g, b])

    center = points.mean(axis=0)
    max_range = max((points.max(axis=0) - points.min(axis=0)).max(), 1.0)

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{ margin: 0; overflow: hidden; background: #000; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white;
                font-family: monospace; background: rgba(0,0,0,0.5); padding: 10px; }}
    </style>
</head>
<body>
    <div id="info">{title}<br>Voxels: {len(points)}<br>Drag to rotate, scroll to zoom</div>
    <script type="importmap">
    {{
        "imports": {{
            "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
            "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
        }}
    }}
    </script>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';

        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x111111);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.01, 100);
        camera.position.set({max_range*2}, {max_range*2}, {max_range*2});

        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.target.set({center[0]}, {center[1]}, {center[2]});
        controls.update();

        // Point cloud with vertex colors
        const positions = new Float32Array({positions});
        const colors = new Float32Array({colors});
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({{
            size: 3.0,
            vertexColors: true,
            sizeAttenuation: false
        }});
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);

        // Axes helper
        const axesHelper = new THREE.AxesHelper({max_range});
        scene.add(axesHelper);

        // Grid
        const gridHelper = new THREE.GridHelper({max_range * 2}, 16, 0x444444, 0x222222);
        scene.add(gridHelper);

        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)

def create_html_visualizations(samples, output_dir, step):
    """Create HTML visualizations for generated samples."""
    import matplotlib.pyplot as plt

    # Print sample statistics
    global_mean = samples.mean()
    print(f"  Sample stats: mean={global_mean:.3f}, std={samples.std():.3f}, min={samples.min():.3f}, max={samples.max():.3f}")

    # Create HTML for first 10 samples (all voxels with intensity-based colors)
    num_html = min(10, len(samples))
    for i in range(num_html):
        voxel = samples[i]  # (16, 16, 16)
        points, intensities = voxels_to_points_with_intensity(voxel)

        html_path = f'{output_dir}/sample_{i:03d}.html'
        create_html_visualization(points, intensities, html_path, title=f"Generated Sample {i} (Step {step})")
        print(f"  Created: sample_{i:03d}.html ({len(points)} voxels, intensity range [{intensities.min():.3f}, {intensities.max():.3f}])")

    # Create grid visualization using matplotlib
    print("\nCreating grid visualization...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Use actual data range for better contrast
    vmin, vmax = samples.min(), samples.max()

    for i, ax in enumerate(axes.flat):
        if i >= len(samples):
            ax.axis('off')
            continue

        voxel = samples[i]
        # Show max projection (best view)
        projection = voxel.max(axis=2)  # Max over depth
        im = ax.imshow(projection, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_title(f'Sample {i}', fontsize=10)
        ax.axis('off')

    plt.suptitle(f"Generated Samples (Step {step}) - Max Projection", fontsize=14)

    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Intensity', shrink=0.8)
    plt.tight_layout()

    grid_path = f'{output_dir}/all_samples_grid.png'
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Created: all_samples_grid.png")

def main():
    # Configuration
    config_path = 'configs/train_3d_v5.yml'
    checkpoint_dir = 'workdir_3d_v5'  # Will use latest checkpoint
    output_dir = 'workdir_3d_v5/generated_samples'
    device_batch_size = 2  # Samples per device (not used, kept for compatibility)

    print("="*60)
    print("3D MNIST Sample Generation")
    print("="*60)
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Output: {output_dir}")
    print(f"Generating: 10 samples (one per digit 0-9)")
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

    # Generate class-specific samples (one per digit 0-9)
    print(f"\nGenerating class-specific samples (digits 0-9)...")
    num_devices = jax.local_device_count()

    all_samples = []
    all_labels = []

    for class_idx in range(10):  # Generate one sample per digit
        print(f"  Generating digit {class_idx}...")
        sample_idx = jnp.arange(num_devices) + class_idx

        # Broadcast class_idx to match sample_idx shape for pmap
        class_idx_array = jnp.full_like(sample_idx, class_idx)

        # Sample with EMA parameters for specific class
        variable = {"params": state.ema_params}
        latent = p_sample_step(variable, sample_idx=sample_idx, class_idx=class_idx_array)
        latent = latent.reshape(-1, *latent.shape[2:])  # Flatten device dimension

        # Transpose to (B, H, W, C) and convert to numpy
        samples = latent.transpose(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        samples = np.array(samples)

        # Take first sample (each device generates device_batch_size samples)
        all_samples.append(samples[0])
        all_labels.append(class_idx)

    # Stack into array
    all_samples = np.stack(all_samples, axis=0)  # Shape: (10, 16, 16, 16)

    print(f"\nGenerated samples shape: {all_samples.shape}")
    print(f"Value range: [{all_samples.min():.3f}, {all_samples.max():.3f}]")

    # Save samples
    import os
    os.makedirs(output_dir, exist_ok=True)

    output_path = f'{output_dir}/samples_{step}.npy'
    np.save(output_path, all_samples)
    print(f"\nSaved samples to: {output_path}")

    # Save individual class-specific samples
    for i in range(10):
        individual_path = f'{output_dir}/sample_{i:03d}.npy'
        np.save(individual_path, all_samples[i])
    print(f"Saved class-specific samples:")
    print(f"  sample_000.npy = digit 0")
    print(f"  sample_001.npy = digit 1")
    print(f"  ...")
    print(f"  sample_009.npy = digit 9")

    # Create interactive HTML visualizations
    print("\n" + "="*60)
    print("Creating interactive HTML visualizations...")
    print("="*60)

    create_html_visualizations(all_samples, output_dir, step)

    print("\n" + "="*60)
    print("Generation complete!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  {output_path}")
    print(f"  {output_dir}/sample_*.npy")
    print(f"  {output_dir}/sample_*.html (interactive 3D views)")
    print(f"  {output_dir}/all_samples_grid.png")
    print(f"\nTo view interactive 3D:")
    print(f"  Open {output_dir}/sample_000.html in your browser!")

if __name__ == '__main__':
    main()
