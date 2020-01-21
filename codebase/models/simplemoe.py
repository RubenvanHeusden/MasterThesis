import torch
import numpy
import torch.nn as nn
import torch.nn.functional as F
from codebase.data.synthdata import *
import numpy as np


def synth_data(num_points, num_features, num_classes):
    X = torch.rand(num_points, num_features)
    y = torch.randint(high=num_classes, size=(num_classes, 1))
    return X, y


class ExpertNetwork(nn.Module):
    # For testing purposes this architecture is fixed for now
    def __init__(self, input_dim, output_dim):
        super(ExpertNetwork, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


class SimpleMoE(nn.Module):
    """
    This class implements a simple Mixture of Model as described in (Jacobs et al. 1991)

    """
    def __init__(self, num_in, gating_network, expert_networks, num_out):
        super(SimpleMoE, self).__init__()
        self.num_in = num_in
        self.gating_network = gating_network
        self.expert_networks = expert_networks
        self.num_out = num_out

    def forward(self, x):

        # the first step is to put x through all the networks
        expert_weights = self.gating_network(x).unsqueeze(1)
        x = torch.stack([net(x) for net in self.expert_networks], dim=0).permute(1, 0, 2)
        x = torch.bmm(expert_weights, x)
        return x


# X, y = synth_data(num_points=128, num_features=3, num_classes=2)
# model = SimpleMoE(num_in=3, gating_network=ExpertNetwork(3, 4), expert_networks=[ExpertNetwork(3, 2) for x in range(4)],
#                   num_out=2)
# model(X)
