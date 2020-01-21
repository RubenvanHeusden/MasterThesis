import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# This file contains a generic training function for training pytorch networks
# model = SimpleMoE(input_dim=3, gating_network=MLP(3, [16], 4), expert_networks=[MLP(3, [8, 16, 8], 2) for x in range(4)],
#                   output_dim=2)
