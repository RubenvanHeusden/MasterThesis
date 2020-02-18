import torch
import torch.nn as nn
from typing import List, Any, Dict
# This class implements a Multigate Mixture of Experts (MMoE) model, as first
# described by (Ma et al. 2018)


class MultiGateMixtureofExperts(nn.Module):
    def __init__(self, shared_layers: List[Any], gating_networks: List[Any], towers: Dict[Any, Any], device,
                 include_lens: bool, batch_size: int):
        """

        @param shared_layers: a list of nn.Modules through which the input is fed

        @param gating_networks: a list of nn.Modules specifying the gating functions
        for each task, the length of this list should be equal to the number of tasks

        @param towers: a list of nn.modules specifying the task specific layers
        the length of this list should be equal to the number of tasks, the input
        dimensions of the modules must be equal to the output dimension of the shared layers
        and the output dimensions should match the number of possible classes for each task

        @param device: torch.device() specifying on which device the model is run

        @param include_lens: Boolean indicating whether to include the lengths of the
        original sequences before padding (mostly for LSTM and RNN models)

        @param batch_size: integer specifying the batch size
        """
        super(MultiGateMixtureofExperts, self).__init__()

        self.gating_networks = nn.ModuleList(gating_networks)
        self.shared_layers = nn.ModuleList(shared_layers)
        self.towers = nn.ModuleList(towers.keys())
        self.tower_dict = {name: x for name, x in zip(towers.values(), range(len(towers)))}
        self.batch_size = batch_size
        self.device = device
        self.include_lens = include_lens
        self.params = {"None": None}
        # Check dimensions here!
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, tower):
        """
        @param x: if include_lens is False, this is a matrix of size [batch_size, max_sent_length]
        containing indices into the vocabulary matrix. If include_lens is True, x should be a tuple
        containing (x: [batch_size, max_sent_length], lengths: [batch_size]) where lengths should contain
        integers indicating the true length of each sequence before padding

        @param tower: a string indicating which task specific tower should be used

        @return: the output of the Multigate Mixture of Experts model
        """
        # Depending on the task we select the appropriate gating network and
        # Task specific tower and compute the activations for that batch
        expert_weights = self.softmax(self.gating_networks[self.tower_dict[tower]](x)).unsqueeze(1)
        x = torch.stack([net(x) for net in self.shared_layers], dim=0).permute(1, 0, 2)
        x = torch.bmm(expert_weights, x)
        x = self.towers[self.tower_dict[tower]](x)
        return x.squeeze(), expert_weights
