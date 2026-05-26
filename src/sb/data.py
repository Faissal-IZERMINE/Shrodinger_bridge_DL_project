"""Data loaders for the MNIST <-> EMNIST experiments (paper §6.2).

Images are resized to 32x32, normalised to [-1, 1], and flattened to length 1024.
EMNIST is filtered to the 10 class subset used in the report.
"""

from __future__ import annotations

import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset


# EMNIST "byclass" target labels used in the report (10 classes).
EMNIST_CLASS_SUBSET = [10, 11, 12, 13, 14, 36, 37, 38, 39, 40]


def _image_transform() -> T.Compose:
    """32x32, [-1, 1], flat 1024-d tensors."""
    return T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize((0.5,), (0.5,)),
        T.Lambda(lambda x: x.flatten()),
    ])


class LawDataset(Dataset):
    """Thin Dataset wrapper that returns the i-th sample of an in-memory tensor."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def _filter_and_remap_emnist(dataset, classes=EMNIST_CLASS_SUBSET):
    mask = torch.isin(dataset.targets, torch.tensor(classes))
    # EMNIST stores its images transposed compared to MNIST; permute.
    dataset.data = dataset.data[mask].permute(0, 2, 1)
    dataset.targets = dataset.targets[mask]
    remap = {old: new for new, old in enumerate(classes)}
    dataset.targets = torch.tensor([remap[t.item()] for t in dataset.targets])
    return dataset


def load_mnist_emnist(subset_size: int = 5000, only_some_letters: bool = True,
                     seed: int = 42, root: str = "./data") -> tuple[torch.Tensor, torch.Tensor]:
    """Load MNIST and EMNIST as paired-size unpaired datasets.

    Returns ``(mnist, emnist)`` as ``(N, 1024)`` float tensors in [-1, 1].
    Class filtering is applied when ``only_some_letters=True`` (default).
    """
    transform = _image_transform()

    mnist_dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform,
    )

    if only_some_letters:
        emnist_dataset = torchvision.datasets.EMNIST(
            root=root, split="byclass", train=True, download=True, transform=transform,
        )
        emnist_dataset = _filter_and_remap_emnist(emnist_dataset)
    else:
        emnist_dataset = torchvision.datasets.EMNIST(
            root=root, split="letters", train=True, download=True, transform=transform,
        )

    g = torch.Generator().manual_seed(seed)
    mnist_size = min(subset_size, len(mnist_dataset))
    emnist_size = min(subset_size, len(emnist_dataset))
    mnist_idx = torch.randperm(len(mnist_dataset), generator=g)[:mnist_size]
    emnist_idx = torch.randperm(len(emnist_dataset), generator=g)[:emnist_size]

    mnist_data = torch.stack([mnist_dataset[i][0] for i in mnist_idx])
    emnist_data = torch.stack([emnist_dataset[i][0] for i in emnist_idx])
    return mnist_data, emnist_data


def get_emnist_test(n_samples: int = 4000, seed: int = 42,
                    root: str = "./data") -> torch.Tensor:
    """EMNIST test split filtered to the same 10-class subset used in training."""
    transform = _image_transform()
    dataset = torchvision.datasets.EMNIST(
        root=root, split="byclass", train=False, download=True, transform=transform,
    )
    dataset = _filter_and_remap_emnist(dataset)

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(dataset), generator=g)[:n_samples]
    return torch.stack([dataset[i][0] for i in idx])


def get_mnist_train_all(root: str = "./data") -> torch.Tensor:
    """All 60K MNIST training images, flattened. Used as FID reference."""
    transform = _image_transform()
    dataset = torchvision.datasets.MNIST(
        root=root, train=True, download=True, transform=transform,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=4,
    )
    return torch.cat([imgs for imgs, _ in loader], dim=0)
