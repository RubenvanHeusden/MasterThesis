import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    This class implements a generic Multi Layer Perceptron with tanh non-linearities
    This class can be used as either a gating function are an expert network in
    the Mixture of Experts classes

    :params
            :input_dim (int)
                specifies the dimensionality of an input sample
            : layer_sizes (list)
                a list of integers specifying the size of each hidden layer, the number of hidden layers is
                inferred from the length of the list.

    """

    def __init__(self, input_dim, layer_sizes, output_dim):
        super(MLP, self).__init__()
        self.input_layer = nn.Linear(input_dim, layer_sizes[0])
        self.fc_layers = nn.ModuleList([nn.Linear(layer_sizes[x], layer_sizes[x+1])
                                        for x in range(len(layer_sizes)-1)])
        self.output_layer = nn.Linear(layer_sizes[-1], output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.tanh(x)
        for layer in self.fc_layers:
            x = layer(x)
            x = self.tanh(x)
        x = self.output_layer(x)
        # return softmax of x depending on loss function
        return x







