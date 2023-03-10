import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Union
import torch.types
from torchvision import transforms


class CustomDataset(Dataset[Any]):
    def __init__(self, folder_path: str, transform: transforms.Compose = None) -> None:
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = os.listdir(folder_path)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Image.Image]:
        image_path = os.path.join(self.folder_path, self.image_paths[idx])
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image
