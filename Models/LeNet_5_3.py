import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, num_class) -> None:
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(5,5), stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=(2,2), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5), stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=6*61*61, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)

    def forward(self, x):
        x = self.pool(nn.functional.sigmoid(self.conv1(x)))
        x = self.pool(nn.functional.sigmoid(self.conv2(x)))
        x = x.view(-1, 6 * 61 * 61)
        x = nn.functional.sigmoid(self.fc1(x))
        x = nn.functional.sigmoid(self.fc2(x))
        x = self.fc3(x)
        x = torch.softmax(x,dim=1)
        return x