import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvNet(nn.Module):
    """
    This class implements a basic convolutional neural network for
    text classification as presented in (Kim,. 2014)

    """
    def __init__(self, input_channels, output_dim, filter_list, embed_matrix, num_filters, dropbout_probs=0.5):
        super(ConvNet, self).__init__()
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=n)
                                      for n in filter_list])

        self.max_over_time_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.fc_layer = nn.Linear(num_filters*len(filter_list), output_dim)
        self.dropout = nn.Dropout(p=dropbout_probs)
        self.embed = nn.Embedding(embed_matrix.shape[0], embedding_dim=300)
        self.embed.weight.data.copy_(embed_matrix)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.embed(x)
        filter_outs = []
        for module in self.filters:
            filter_outs.append(self.max_over_time_pool(module(x)))
        pen_ultimate_layer = torch.cat(filter_outs, dim=1)
        output = self.dropout(pen_ultimate_layer).squeeze()
        output = self.fc_layer(output)

        return output


