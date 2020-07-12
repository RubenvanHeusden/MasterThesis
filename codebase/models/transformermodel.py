import torch
import torch.nn as nn
import sys

sys.path.append('/content/gdrive/My Drive')
from codebase.models.positionalencoder import PositionalEncoder
import math


# See if I can update PyTorch so it doesn't get angry about not finding Transformers

# TODO: check if I am doing the right amount of linear layers and dropout order
# TODO check the positional encoder workings to make sure its doing what
# I think it should be doing


class TransformerModel(nn.Module):
    def __init__(self, max_seq_len: int, num_outputs: int, word_embedding_matrix, feed_fwd_dim: int = 2048,
                 num_transformer_layers=6, num_transformer_heads=6, pos_encoding_dropout: float = 0.1,
                 classification_dropout: float = 0.2, batch_first: bool = False, pad_index: int = 1,
                 is_gate=False, use_bert_embeds=False):
        """

        :param max_seq_len: integer specifying the maximum length of the input sequences.
        sequences longer than max_seq_len will be truncated, sentenced shorter than
        max_seq_len will be padded

        :param num_outputs: integer specifying the size of the output of the final linear layer,
        for sequence classification this is equal to the number of classes

        :param word_embedding_matrix: matrix of size (vocab_size, embedding_dim) where each row
        represents the embedding for a specific word

        :param feed_fwd_dim: integer specifying the size of the feedforward layers used inside
        the transformer encoder layers

        :param num_transformer_layers: integer specifying the number of transformer Modules used
        in the architecture. default is 6

        :param num_transformer_heads: Integer specifying the number of attention heads in the Transformer
        default is 6.

        :param pos_encoding_dropout: float specifying the dropout probability for the positional encodings
        used when adding periodic information to the word embeddings

        :param classification_dropout: float specifying the dropout probability before applying the
        final linear layer

        :param batch_first: Boolean specifying whether the batch dimension is the first dimension of the input
        or not. when True the expected input size is (batch_size, max_seq_len), when False the expected size
        is (max_seq_len, batch_size)

        :param pad_index: Integer specifying the number that is used to represent padding tokens.
        This is used to mask the padded tokens when the input is fed through the Transformer
        """
        super(TransformerModel, self).__init__()

        self.max_seq_len = max_seq_len
        self.num_outputs = num_outputs
        self.batch_first = batch_first
        self.embedding_dim = word_embedding_matrix.shape[1]
        self.embed_matrix = nn.Embedding(*word_embedding_matrix.shape)
        self.embed_matrix.weight.data.copy_(word_embedding_matrix)
        self.pad_index = pad_index

        self.positional_encoder = PositionalEncoder(max_seq_len, embedding_dim=word_embedding_matrix.shape[1],
                                                    dropout=pos_encoding_dropout, batch_first=batch_first)

        self._transform_encoder_layer = nn.TransformerEncoderLayer(d_model=word_embedding_matrix.shape[1],
                                                                   nhead=num_transformer_heads,
                                                                   dim_feedforward=feed_fwd_dim)

        self.transformer = nn.TransformerEncoder(self._transform_encoder_layer, num_layers=num_transformer_layers)
        self.dropout = nn.Dropout(p=classification_dropout)
        self.final_fc = nn.Linear(word_embedding_matrix.shape[1], num_outputs)
        self.init_weights()
        self.is_gate = is_gate
        self.use_bert_embeddings = use_bert_embeds

    def forward(self, x):
        if isinstance(x, list):
            x = x[0]
        if self.use_bert_embeddings:
            x = self.positional_encoder(x)
            x = self.transformer(x)
            return self.final_fc(self.dropout(x[0, :, :]))

        if self.is_gate:
            init_tokens = torch.tensor([2 for _ in range(x.shape[0])]).unsqueeze(1).cuda()
            x = torch.cat((init_tokens, x), dim=1)

        pad_mask = torch.zeros_like(x)
        pad_mask[x == self.pad_index] = 1
        if self.batch_first:
            assert x.shape[1] == self.max_seq_len, " max_seq_len is of incorrect size for batch_first = True." \
                                                   " Expected %d, got %d" % (self.max_seq_len, x.shape[1])
            pad_mask = pad_mask.bool()
        else:
            assert x.shape[0] == self.max_seq_len, " max_seq_len is of incorrect size for batch_first = False." \
                                                   " Expected %d, got %d" % (self.max_seq_len, x.shape[0])
            pad_mask = pad_mask.transpose(0, 1).bool()

        x = self.embed_matrix(x)
        x = self.positional_encoder(x)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        # grab the cls token hidden dim for the linear output layer
        return self.final_fc(self.dropout(x[0, :, :]))

    def init_weights(self):
        initrange = 0.1
        self.final_fc.bias.data.zero_()
        self.final_fc.weight.data.uniform_(-initrange, initrange)
