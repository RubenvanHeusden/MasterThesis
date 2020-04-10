import torch
import torch.nn as nn
import math


class PositionalEncoder(nn.Module):
    def __init__(self, max_seq_len: int, embedding_dim: int, dropout: float = 0.1, batch_first=False):
        """
        :param max_seq_len: Integer specifying the maximum sequence length of the input examples
        in a batch

        :param embedding_dim: Integer specifying the size of the embeddings used

        :param dropout: float specifying the dropout probability after applying the encoding

        :param batch_first: Boolean specifying whether the first dimension is the batch_size, default
        is False.
        """
        super(PositionalEncoder, self).__init__()
        positional_encodings = torch.zeros(max_seq_len, embedding_dim)
        positions = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        encoding_multiplier = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))

        positional_encodings[:, 0::2] = torch.sin(positions * encoding_multiplier)
        positional_encodings[:, 1::2] = torch.cos(positions * encoding_multiplier)

        positional_encodings = positional_encodings.unsqueeze(0).transpose(0, 1)

        self.register_buffer('positional_encodings', positional_encodings)
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim

    def forward(self, x):
        """
        :param x: Tensor containing the (padded) sequence of word embeddings. When batch_first is False,
        the input is expected to have the shape [max_seq_len, batch_size, embedding_dim]. If batch_first
        is True, the input is expected to have the shape [batch_size, max_seq_len, embedding_dim]

        :return: Torch.Tensor of shape [max_seq_len, batch_size, embedding_dim] with the positional information
        add to the word representation. Because of the expected shape of the Tensor for the Transformer module
        the output shape is [max_seq_len, batch_size, embedding_dim], regardless of the value of batch_first
        """
        assert x.shape[-1] == self.embedding_dim, "the embedding size is not correct. expected %d, got %d" \
                                                  % (self.embedding_dim, x.shape[-1])
        if self.batch_first:
            assert x.shape[1] == self.max_seq_len, " max_seq_len is of incorrect size for batch_first = True." \
                                                   " Expected %d, got %d" % (self.max_seq_len, x.shape[1])
            x = x + (self.positional_encodings[:x.size(1)].permute(1, 0, 2))
            x = x.permute(1, 0, 2)
        else:
            assert x.shape[0] == self.max_seq_len, " max_seq_len is of incorrect size for batch_first = False." \
                                                   " Expected %d, got %d" % (self.max_seq_len, x.shape[0])
            x = x + self.positional_encodings[:x.size(0), :]
        return self.dropout(x)