"""3D-MNIST dataset loader for JAX/Flax."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class MNIST3DDataset(Dataset):
    """3D-MNIST dataset loader.

    Loads 64x64 grayscale images from the full_dataset_vectors.h5 file.
    """

    def __init__(self, data_path, split='train'):
        """
        Args:
            data_path: Path to the directory containing full_dataset_vectors.h5
            split: 'train' or 'test'
        """
        h5_file = f"{data_path}/full_dataset_vectors.h5"

        with h5py.File(h5_file, 'r') as f:
            if split == 'train':
                self.images = f['X_train'][:]
                self.labels = f['y_train'][:]
            else:
                self.images = f['X_test'][:]
                self.labels = f['y_test'][:]

        # Reshape from (N, 4096) to (N, 64, 64, 1) and convert to uint8 [0, 255]
        self.images = self.images.reshape(-1, 64, 64, 1)
        self.images = (self.images * 255).astype(np.uint8)

        print(f"Loaded {split} set: {len(self.images)} images, shape={self.images.shape}")
        print(f"Label distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Return a single sample.

        Returns:
            dict with 'image' (64, 64, 1) uint8 and 'label' int
        """
        return {
            'image': self.images[idx],
            'label': int(self.labels[idx])
        }


def create_mnist3d_loader(data_path, batch_size, split='train', num_workers=4):
    """Create a DataLoader for 3D-MNIST.

    Args:
        data_path: Path to the directory containing full_dataset_vectors.h5
        batch_size: Batch size
        split: 'train' or 'test'
        num_workers: Number of worker processes

    Returns:
        dataloader, num_samples, steps_per_epoch
    """
    dataset = MNIST3DDataset(data_path, split=split)
    num_samples = len(dataset)

    # Use DistributedSampler for multi-process training if needed
    try:
        import jax
        if jax.process_count() > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=(split == 'train')
            )
            shuffle = False
        else:
            sampler = None
            shuffle = (split == 'train')
    except:
        sampler = None
        shuffle = (split == 'train')

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    steps_per_epoch = len(dataloader)

    return dataloader, num_samples, steps_per_epoch
