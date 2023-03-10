import torch
from typing import Any, Tuple
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset[Any]):
    def __init__(self, images: list[str], labels: list[int], transform: transforms.Compose = None) -> None:
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.load(self.images[index]), torch.tensor(self.labels[index])
