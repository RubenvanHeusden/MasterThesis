# This class implements a general structure for a multitask model
# it expects both a shared  bottom layer or network which is shared by
# all tasks, as well as  task-specific 'towers' that are used for  the
# specific tasks.

import torch.nn as nn
from typing import Any, Dict


class MultiTaskModel(nn.Module):
    def __init__(self, shared_layer, towers: Dict[Any, Any], input_dimension: int, batch_size: int,
                 device, include_lens: bool = True):

        super(MultiTaskModel, self).__init__()
        self.shared_layer = shared_layer
        self.tower_list = nn.ModuleList(towers.keys())
        self.tower_dict = {name: x for name, x in zip(towers.values(), range(len(towers)))}
        self.input_dimension = input_dimension
        self.batch_size = batch_size
        self.device = device
        self.include_lens = include_lens
        self.params = {"None": None}

    def forward(self, x, tower="category"):
        x = self.shared_layer(x)
        x = self.tower_list[self.tower_dict[tower]](x)
        # TODO maybe add a softmax
        return x
