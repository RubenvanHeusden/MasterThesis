import torch
import torch.nn as nn
from typing import List


class MLPGate(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, embedding_matrix):
        """
        This class implements a generic Multi Layer Perceptron with tanh non-linearities
        This class can be used as either a gating function are an expert network in
        the Mixture of Experts classes

        :param input_dim: integer specifying the dimensionality of an input sample

        :param output_dim: integer specifying the dimensionality of the output

        :param embedding_matrix: pytorch matrix of size [vocab_size X word_embedding_dim] where each row
        signifies a word embedding vector
        """
        super(MLPGate, self).__init__()
        self.hidden = nn.Linear(input_dim*embedding_matrix.shape[1], output_dim)
        self.softmax = nn.Softmax(dim=1)
        self.embed = nn.Embedding(*embedding_matrix.shape)
        self.embed.weight.data.copy_(embedding_matrix)

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = self.embed(x)
        x = x.reshape(x.shape[0], -1)
        out = self.softmax(self.hidden(x))
        return out
