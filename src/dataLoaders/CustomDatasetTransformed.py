from typing import Any, Tuple, List

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset[Any]):
    def __init__(self, images: List[str], labels: List[int], transform: Any = None) -> None:
        """
        Initialize the CustomDataset class.

        Args:
            images (List[str]): List of image paths.
            labels (List[int]): List of corresponding labels.
            transform (transforms.Compose, optional): Data transformation to be applied to the images. Defaults to None.
        """
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """
        Return the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve the item at the given index from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the image and its label.
        """
        return torch.load(self.images[index]), torch.tensor(self.labels[index])
