import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskConvNet(nn.Module):
    """
    This class implements a basic convolutional neural network for
    text classification as presented in (Kim,. 2014)

    Because this model is used in the multitask setting, the output of the model is not
    fed through a linear layer to accomodate for different task output size.
    """
    def __init__(self, input_channels, filter_list, embed_matrix, num_filters, dropbout_probs=0.5,
                 use_bert_embeds: bool = False):
        super(MultitaskConvNet, self).__init__()
        self.params = locals()
        self.input_channels = input_channels
        self.filters = nn.ModuleList([nn.Conv2d(in_channels=1, out_channels=num_filters,
                                                kernel_size=(n, embed_matrix.shape[1])) for n in filter_list])

        self.max_over_time_pool = torch.nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.dropout = nn.Dropout(p=dropbout_probs)
        self.embed = nn.Embedding(*embed_matrix.shape)
        self.embed.weight.data.copy_(embed_matrix)
        self.relu = nn.ReLU()
        self.use_bert_embeddings = use_bert_embeds

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        x = x.unsqueeze(1)
        if not self.use_bert_embeddings:
            x = self.embed(x)
        filter_outs = []
        for module in self.filters:
            filter_outs.append(self.max_over_time_pool(self.relu(module(x))))
        pen_ultimate_layer = torch.cat(filter_outs, dim=1)
        output = self.dropout(pen_ultimate_layer).squeeze()
        return output

