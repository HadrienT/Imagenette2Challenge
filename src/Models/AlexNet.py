import torch
import torch.nn as nn

# class Model(nn.Module):
#     def __init__(self,num_class) -> None:
#         super(Model, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=(11,11),stride=4,padding=2)
#         self.pool = nn.MaxPool2d(kernel_size=(3,3),stride=2)
#         self.conv2 = nn.Conv2d(in_channels=64,out_channels=192,kernel_size=(5,5),stride=1,padding=2)
#         self.conv3 = nn.Conv2d(in_channels=192,out_channels=384,kernel_size=(3,3),stride=1,padding=1)
#         self.fc1 = nn.Linear(in_features=16*59*59,out_features=4096)
#         self.fc2 = nn.Linear(in_features=4096,out_features=4096)
#         self.fc3 = nn.Linear(in_features=4096,out_features=num_class)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = self.pool(x)
#         x = torch.relu(self.conv2(x))
#         for _ in range(1):
#             x = torch.relu(self.conv3(x))
#         x = self.pool(x)
#         x = x.view(-1,16*59*59)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x


class Model(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(Model, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
