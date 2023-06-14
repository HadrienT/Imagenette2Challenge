from typing import Any, Tuple, Union, List
import PIL.Image as Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class CustomDataset(Dataset[Any]):
    def __init__(self, image_paths: List[str], labels: List[int], transform: transforms.Compose = None) -> None:
        """
        Initialize the CustomDataset class.

        Args:
            image_paths (List[str]): List of image paths.
            labels (List[int]): List of corresponding labels.
            transform (transforms.Compose, optional): Data transformation to be applied to the images. Defaults to None.
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[Union[torch.Tensor, Image.Image], torch.Tensor]:
        """
        Retrieve the item at the given index from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[Union[torch.Tensor, Image.Image], torch.Tensor]: Tuple containing the image and its label.
        """
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index])
