import torch
import torch.nn as nn
from typing import List, Any, Dict
# This class implements a Multigate Mixture of Experts (MMoE) model, as first
# described by (Ma et al. 2018)


def average_out_weights_stateful(weights, running_mean, m):
    running_assignments = running_mean.expand(weights.shape[0], -1)
    sum_assignments = torch.add(running_assignments, weights.cumsum(dim=0))
    average_assignments = sum_assignments.mean(dim=1).unsqueeze(1)
    mask = (sum_assignments - average_assignments.expand(-1, weights.shape[1])) <= m
    corrected_weights = torch.mul(weights, mask)
    updated_running_mean = sum_assignments[-1]
    return corrected_weights, updated_running_mean


def average_out_weights(weights, m):
    sum_assignments = weights.cumsum(dim=0)
    average_assignments = sum_assignments.mean(dim=1).unsqueeze(1)
    mask = (sum_assignments - average_assignments.expand(-1, weights.shape[1])) <= m
    corrected_weights = torch.mul(weights, mask)
    return corrected_weights


class MultiGateMixtureofExperts(nn.Module):
    def __init__(self, shared_layers: List[Any], gating_networks: List[Any], towers: Dict[Any, Any], device,
                 include_lens: bool, batch_size: int, return_weights: bool = True, gating_drop=0.1,
                 mean_diff=0.1, weight_adjust_mode="reassign_mean_stateful"):
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
        self.gating_net_running_means = {tower: torch.zeros(size=(1, len(shared_layers))).to(device) for tower in towers.values()}

        self.towers = nn.ModuleList(towers.keys())
        self.tower_dict = {name: x for name, x in zip(towers.values(), range(len(towers)))}
        self.batch_size = batch_size
        self.device = device
        self.include_lens = include_lens
        self.params = {"None": None}
        self.softmax = nn.Softmax(dim=1)
        self.return_weights = return_weights
        self.gating_drop = nn.Dropout(p=gating_drop)
        self.weight_adjust_mode = weight_adjust_mode
        self.m = mean_diff

        print("\n")
        print("\t - MMoE Model Settings - \t")
        print("-> Expert Network Architecture: ")
        print("-> tower network structure : ")
        print("-> Expert Balancing Strategy: %s" % self.weight_adjust_mode)
        if weight_adjust_mode == "reassign_mean" or weight_adjust_mode == "reassign_mean_stateful":
            print("-> mean difference treshold: %.2f" % self.m)
        if weight_adjust_mode == "dropout":
            print("-> Expert dropout ratio: %.2f" % self.gating_drop.p)

    def forward(self, x, tower=("category", "emotion")):
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
        stacked_x = torch.stack([net(x) for net in self.shared_layers], dim=0).permute(1, 0, 2)

        outputs = []
        weights = []
        for t in tower:
            if self.weight_adjust_mode == "reassign_mean":
                raw_weights = self.softmax(self.gating_networks[self.tower_dict[t]](x))
                expert_weights = self.softmax(average_out_weights(raw_weights, self.m)).unsqueeze(1)

            elif self.weight_adjust_mode == "reassign_mean_stateful":
                raw_weights = self.softmax(self.gating_networks[self.tower_dict[t]](x))
                expert_weights, new_mean = average_out_weights_stateful(raw_weights,
                                                                        self.gating_net_running_means[t], self.m)
                self.gating_net_running_means[t] = new_mean
                expert_weights = self.softmax(expert_weights).unsqueeze(1)
            elif self.weight_adjust_mode == "dropout":
                expert_weights = self.softmax(self.gating_drop(self.softmax(self.gating_networks[self.tower_dict[t]](x)))).unsqueeze(1)
            else:
                expert_weights = self.softmax(self.gating_networks[self.tower_dict[t]](x)).unsqueeze(1)

            weighted_x = torch.bmm(expert_weights, stacked_x)
            print(weighted_x.shape)
            quit()
            weighted_x = self.towers[self.tower_dict[t]](weighted_x)
            outputs.append(weighted_x.squeeze())
            weights.append(expert_weights)
        if self.return_weights:
            return outputs, weights
        return outputs
