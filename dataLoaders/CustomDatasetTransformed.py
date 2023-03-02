import torch.utils.data as data
import torch 

class CustomDataset(data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return torch.load(self.images[index]),torch.tensor(self.labels[index])
