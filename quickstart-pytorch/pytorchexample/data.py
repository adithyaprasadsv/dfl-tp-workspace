from __future__ import annotations
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter

def generate_distributed_datasets(k: int, alpha: float, save_dir: str) -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Load FashionMNIST training set
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.FashionMNIST(
        root="./raw_data", train=True, download=True, transform=transform
    )

    # Group indices by class label
    labels = np.array(full_dataset.targets)   # shape: (60000,)
    num_classes = 10
    class_indices = [np.where(labels == c)[0] for c in range(num_classes)]

    # For each class, sample Dirichlet proportions across K clients
    client_indices = [[] for _ in range(k)] 

    rng = np.random.default_rng(42)

    for c in range(num_classes):
        indices_c = class_indices[c].copy()
        rng.shuffle(indices_c)

        proportions = rng.dirichlet(alpha=np.repeat(alpha, k))
        splits = (np.cumsum(proportions) * len(indices_c)).astype(int)
        splits = np.clip(splits, 0, len(indices_c))

        prev = 0
        for i, end in enumerate(splits):
            client_indices[i].extend(indices_c[prev:end].tolist())
            prev = end

    # print("BEFORE FIX:")
    # for i in range(k):
    #     labels_i = [labels[j] for j in client_indices[i]]
    #     dist = Counter(labels_i)
    #     print(f"  client_{i}: {dict(sorted(dist.items()))}, total={len(labels_i)}")

    # fix clients with too few samples
    min_samples = 64

    for i in range(k):
        deficit = min_samples - len(client_indices[i])
        if deficit <= 0:
            continue

        # Find this client's dominant class
        client_labels = labels[client_indices[i]]
        if len(client_labels) > 0:
            dominant_class = np.bincount(client_labels).argmax()
        else:
            dominant_class = rng.integers(0, num_classes)

        # Sample additional indices from that class with replacement
        class_pool = class_indices[dominant_class]
        extra = rng.choice(class_pool, size=deficit, replace=True).tolist()
        client_indices[i].extend(extra)

    print("AFTER FIX:")
    for i in range(k):
        labels_i = [full_dataset.targets[j].item() for j in client_indices[i]]
        dist = Counter(labels_i)
        print(f"  client_{i}: {dict(sorted(dist.items()))}, total={len(labels_i)}")

    # Save each client's data
    print(f"\nGenerating {k} client partitions (alpha={alpha}):")
    for i in range(k):
        idx = torch.tensor(client_indices[i], dtype=torch.long)
        torch.save(idx, os.path.join(save_dir, f"client_{i}.pt"),)
        # print(f"client_{i}.pt: {len(idx)} samples")

    empty = sum(1 for c in client_indices if len(c) == 0)
    if empty:
        print(f"WARNING: {empty} clients still have 0 samples after fix!")
    else:
        print(f"\nAll {k} clients have >= {min_samples} samples.")


def load_client_data(cid: int, data_dir: str, batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load full dataset
    full_dataset = datasets.FashionMNIST(
        root="./raw_data", train=True, download=False, transform=transform
    )

    # Load saved indices for this client
    idx = torch.load(os.path.join(data_dir, f"client_{cid}.pt"), weights_only=True)
    idx = idx.tolist()

    # check class distribution
    labels = [full_dataset.targets[i].item() for i in idx]
    dist = Counter(labels)
    print(f"  Client {cid} class distribution: {dict(sorted(dist.items()))}")

    # 80/20 split
    split = int(0.8 * len(idx))
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader