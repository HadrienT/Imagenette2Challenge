import PIL.Image as Image
import torch 
from torch.utils.data import Dataset
from typing import Any
class CustomDataset(Dataset[Any]):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(self.labels[index])

