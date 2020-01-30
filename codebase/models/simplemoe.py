import torch
import numpy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from codebase.data_classes.synthdata import *
from codebase.models.mlp import *


class SimpleMoE(nn.Module):
    """
    This class implements a simple Mixture of Model as described in (Jacobs et al. 1991)
    :params
        input_dim: (int
            dimensionality of the input sample(s)

        gating_network: (torch.nn.Module)
            a pytorch Module that takes the input sample(s) and outputs
            a distribution over experts

        expert_networks: (list)
            list of Pytorch Modules acting as the expert networks
            takes the input sample and outputting a prediction of size output_dim

        output_dim: (int)
            dimensionality of the output i.e. the number of classes
            in classification
    """
    def __init__(self, input_dim, gating_network, expert_networks, output_dim, device=torch.device("cpu")):
        super(SimpleMoE, self).__init__()
        self.input_dim = input_dim
        self.gating_network = gating_network
        self.expert_networks = nn.ModuleList(expert_networks)
        for x in range(len(expert_networks)):
            self.expert_networks[x] = self.expert_networks[x].to(device)
        self.output_dim = output_dim
        self.softmax = nn.Softmax(dim=1)
        self.params = {
            "input_dim": input_dim,
            "gating_network": gating_network,
            "expert_networks": expert_networks,
            "output_dim": output_dim,
            "device": device
        }

    def forward(self, x):
        """
        :params
            :x
                input to the neural network, can either be a single sample or
                a batch of shape [batch_size, input_dim]

        feeds the input sample(s) through the expert networks and multiplies the outputs
        by the outputs of the gating network with the outputs of the gating network
        """

        # the first step is to put x through all the networks
        expert_weights = self.softmax(self.gating_network(x)).unsqueeze(1)
        x = torch.stack([net(x) for net in self.expert_networks], dim=0).permute(1, 0, 2)
        x = torch.bmm(expert_weights, x)
        return x.squeeze()



