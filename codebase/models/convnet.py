import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    This class implements a basic convolutional neural network for use
    as an expert network or gating function

    """
    def __init__(self, input_channels, output_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=6, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, output_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
