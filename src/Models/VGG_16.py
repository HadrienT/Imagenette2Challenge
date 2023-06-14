import torch.nn as nn
import torch


class Model(nn.Module):
    """
    Implements the AlexNet model architecture.

    Based on the paper: https://paperswithcode.com/method/vgg-16
    """
    def __init__(self, num_class: int) -> None:
        super(Model, self).__init__()
        self.conv1_bloc1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bloc1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=1)
        self.conv1_bloc2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv2_bloc2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.conv3_bloc2 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.fc1 = nn.Linear(in_features=4096, out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=num_class)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        for _ in range(2):
            x = torch.relu(self.conv1_bloc1(x))
            x = torch.relu(self.conv2_bloc1(x))
            x = self.pool(x)

        for _ in range(3):
            x = torch.relu(self.conv1_bloc2(x))
            x = torch.relu(self.conv2_bloc2(x))
            x = torch.relu(self.conv3_bloc2(x))
            x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc1(x))
        x = self.fc3(x)

        return x
