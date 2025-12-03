#!/usr/bin/env python3
"""Extract generated samples from numpy array and save as images."""

import numpy as np
import os
from PIL import Image
from tqdm import tqdm


# Configuration
samples_file = 'logs/eval_mf_b4/samples/samples_all.npy'
output_dir = 'logs/eval_mf_b4/extracted_images'
num_samples = 10  # Number of individual images to save
create_grid = True  # Set to True to create grid visualization
grid_file = 'logs/eval_mf_b4/sample_grid.png'
grid_size = (2, 5)  # Rows, columns


def extract_samples():
    """Extract samples from numpy array and save as PNG images."""
    print(f"Loading samples from {samples_file}...")
    samples = np.load(samples_file, mmap_mode='r')
    print(f"Loaded samples shape: {samples.shape}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract and save samples
    num_to_save = min(num_samples, len(samples))
    print(f"Saving {num_to_save} images to {output_dir}...")

    for i in tqdm(range(num_to_save)):
        img_array = samples[i]  # Shape: (256, 256, 3)

        # Convert to PIL Image
        img = Image.fromarray(img_array.astype(np.uint8))

        # Save as PNG
        img_path = os.path.join(output_dir, f'sample_{i:05d}.png')
        img.save(img_path)

    print(f"Done! Saved {num_to_save} images to {output_dir}")


def create_sample_grid():
    """Create a grid visualization of samples."""
    print(f"Loading samples from {samples_file}...")
    samples = np.load(samples_file, mmap_mode='r')

    rows, cols = grid_size
    num_images = rows * cols

    # Get random samples
    indices = np.random.choice(len(samples), size=min(num_images, len(samples)), replace=False)

    # Image size
    img_h, img_w = samples.shape[1:3]

    # Create grid canvas
    grid_h = rows * img_h
    grid_w = cols * img_w
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    print(f"Creating {rows}x{cols} grid...")
    for idx, sample_idx in enumerate(indices):
        row = idx // cols
        col = idx % cols

        y1 = row * img_h
        y2 = y1 + img_h
        x1 = col * img_w
        x2 = x1 + img_w

        grid[y1:y2, x1:x2] = samples[sample_idx]

    # Save grid
    grid_img = Image.fromarray(grid)
    os.makedirs(os.path.dirname(grid_file), exist_ok=True)
    grid_img.save(grid_file)
    print(f"Saved grid to {grid_file}")


if __name__ == '__main__':
    # Extract individual samples
    extract_samples()

    # Create grid visualization
    if create_grid:
        create_sample_grid()
