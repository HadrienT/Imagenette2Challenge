import torch.utils.data as data
import pandas as pd 
import torch

class MNIST_dataset(data.Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path,skiprows=1,header=None)
        self.labels = self.data.iloc[:, 0].values
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data.iloc[index, 1:].values.astype('float32')
        # if self.transform:
        #     data = self.transform(torch.tensor(data))
        target = self.labels[index]
        return data, target

