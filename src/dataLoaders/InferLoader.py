import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Any

class CustomDataset(Dataset[Any]):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = os.listdir(folder_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_paths[idx])
        image = Image.open(image_path)
        
        if self.transform:
            image = self.transform(image)
        
        return image