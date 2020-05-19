import torch
import torch.nn as nn
from typing import List


class MLPGate(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        """
        This class implements a generic Multi Layer Perceptron with tanh non-linearities
        This class can be used as either a gating function are an expert network in
        the Mixture of Experts classes

        @param input_dim: integer specifying the dimensionality of an input sample

        @param layer_sizes:a list of integers specifying the size of each hidden layer, the number of hidden layers is
                inferred from the length of the list.

        @param output_dim: integer specifying the dimensionality of the output
        """
        super(MLPGate, self).__init__()
        self.hidden = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.mean
        return x
