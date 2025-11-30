"""MNIST dataset loader for JAX/Flax."""

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def create_mnist_loader(data_path, batch_size, split='train', num_workers=4):
    """Create a DataLoader for MNIST.

    Args:
        data_path: Path to download/load MNIST data
        batch_size: Batch size
        split: 'train' or 'test'
        num_workers: Number of worker processes

    Returns:
        dataloader, num_samples, steps_per_epoch
    """
    is_train = (split == 'train')

    # Download and load MNIST
    # MNIST images are 28x28 grayscale, we'll resize to 32x32 for better patch division
    transform = transforms.Compose([
        transforms.Resize(32),  # Resize to 32x32 (divisible by 2, 4, 8, 16)
        transforms.ToTensor(),  # Converts to [0, 1] and (C, H, W) format
        transforms.Lambda(lambda x: (x * 255).to(torch.uint8)),  # Convert to uint8 [0, 255]
        transforms.Lambda(lambda x: x.permute(1, 2, 0)),  # (C, H, W) -> (H, W, C)
    ])

    dataset = datasets.MNIST(
        root=data_path,
        train=is_train,
        download=True,
        transform=transform
    )

    num_samples = len(dataset)
    print(f"Loaded MNIST {split} set: {num_samples} samples")

    # Use DistributedSampler for multi-process training if needed
    try:
        import jax
        if jax.process_count() > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=is_train
            )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train
    except:
        sampler = None
        shuffle = is_train

    # Custom collate function to return tuple format (image, label)
    # Images are already (H, W, C), permute to (C, H, W) for prepare_batch_data
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])  # (B, H, W, C)
        images = images.permute(0, 3, 1, 2)  # (B, C, H, W)
        labels = torch.tensor([item[1] for item in batch])
        return (images, labels)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    steps_per_epoch = len(dataloader)

    return dataloader, num_samples, steps_per_epoch
