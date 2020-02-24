import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from typing import List, Any


class ConvNet(nn.Module):
    """
    This class implements a basic convolutional neural network for
    text classification as presented in (Kim,. 2014)

    """
    def __init__(self, input_channels: int, output_dim: int, filter_list: List[int],
                 embed_matrix, num_filters: int, dropbout_probs: float=0.5):
        """

        @param input_channels: integer specifying the number of input channels of the 'image'

        @param output_dim: integer specifying the number of outputs

        @param filter_list: list of integers specifying the size of each kernel that is applied to the image,
        the outputs of the kernels are concatenated to form the input to the linear layer

        @param embed_matrix: torch Tensor with size [size_vocab, embedding_dim] where each row is a
        word embedding

        @param num_filters: the amount of filter to apply to the image, this also determins the size
        of the input to the fully connected layers, which is equal to num_kernels*num_filters

        @param dropbout_probs: float specifying the dropout probabilities
        """
        super(ConvNet, self).__init__()
        self.params = locals()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters,
                                                kernel_size=(n, embed_matrix.shape[1])) for n in filter_list])

        self.max_over_time_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.fc_layer = nn.Linear(num_filters*len(filter_list), output_dim)
        self.dropout = nn.Dropout(p=dropbout_probs)
        self.embed = nn.Embedding(*embed_matrix.shape)
        self.embed.weight.data.copy_(embed_matrix)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        @param x: input of size (batch_size, max_sen_length_batch)
        @return: output of the CNN applied to the input x
        """
        x = x.unsqueeze(1)
        x = self.embed(x)
        filter_outs = []
        for module in self.filters:
            module_out = self.relu(module(x))
            module_out = self.max_over_time_pool(module_out)
            filter_outs.append(module_out)
        pen_ultimate_layer = torch.cat(filter_outs, dim=1)
        output = self.dropout(pen_ultimate_layer).squeeze()
        output = self.fc_layer(output)

        return output
