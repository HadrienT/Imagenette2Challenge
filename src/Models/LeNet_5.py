import torch.nn as nn
import torch
from typing import Tuple


class Model(nn.Module):
    def __init__(self, num_class: int, loss_function: torch.nn.modules.loss) -> None:
        super(Model, self).__init__()

        hidden_size_1 = 6
        hidden_size_2 = 16
        hidden_size_3 = 16 * 61 * 61
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_size_1, kernel_size=(5, 5), stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=hidden_size_1, out_channels=hidden_size_2, kernel_size=(5, 5), stride=1, padding=0)
        self.fc1 = nn.Linear(in_features=hidden_size_3, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_class)
        self.loss = loss_function

    def forward(self, x: torch.Tensor, target: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = torch.tanh(x)
        x = self.pool(x)

        x = x.flatten(start_dim=1)

        x = self.fc1(x)
        x = torch.tanh(x)

        x = self.fc2(x)
        x = torch.tanh(x)

        x = self.fc3(x)

        if self.training:
            loss = self.loss(x, target)
            return x, loss
        return x
