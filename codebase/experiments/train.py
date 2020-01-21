import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from codebase.data.synthdata import SynthData
from codebase.models.simplemoe import SimpleMoE
from codebase.models.mlp import MLP
# This file contains a generic training function for training pytorch networks

# create the dataset
num_datapoints = 1024
num_classes = 5
num_features = 12
num_experts = 3

synthetic_data = SynthData(num_points=num_datapoints, num_classes=num_classes,
                           num_features=num_features)

dataloader = DataLoader(SynthData(248, 10, 3), batch_size=4,
                        shuffle=True, num_workers=0) # weird error when setting num_workers > 0

g = MLP(num_features, [16, 32, 16], num_experts)
experts = [MLP(num_features, [64, 128, 64], num_classes)]

model = SimpleMoE(input_dim=num_features, gating_network=g, expert_networks=experts,
                  output_dim=num_classes)


def train():
    pass
