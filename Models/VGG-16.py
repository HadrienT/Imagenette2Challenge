import torch.nn as nn
import torch
class Model(nn.Module):
    def __init__(self, num_class) -> None:
        super(Model,self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1)