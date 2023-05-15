from typing import Any, Tuple, Union, List
import PIL.Image as Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset[Any]):
    def __init__(self, image_paths: List[str], labels: List[int], transform: transforms.Compose = None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Image.Image], torch.Tensor]:
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index])
