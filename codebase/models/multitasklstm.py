# This class is identical to the simplelstm model with only differnce being the absence
# of a final linear layer in this model, as it should support variable output shapes
import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: add Xavier Initialization
# TODO: set trainable weights to false


class MultiTaskLSTM(nn.Module):
    def __init__(self, vocab, hidden_dim: int, dropout: float = 0.3, device=torch.device("cpu"),
                 use_lengths=True):
        """
        :param vocab: a vector containing the word embeddings of the word in the train set
        :param hidden_dim: int specifying number of hidden units in LSTM
        :param dropout: float specifying the dropout rotia
        :param device: torch.device specifying if model is ran on cpu or gpu
        :param use_lengths: boolean specifying whether to remove padding for LSTM input or not
        """
        super(MultiTaskLSTM, self).__init__()
        self.params = locals()
        self.embed = nn.Embedding(*vocab.shape)
        self.embed.weight.data.copy_(vocab)
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(vocab.shape[1], hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.use_lengths = use_lengths
        self.params = {"vocab": vocab,
                       "embedding_dim": vocab.shape[1],
                       "hidden_dim": hidden_dim,
                       "dropout": dropout,
                       "device": device}

    def forward(self, x):
        if self.use_lengths:
            inputs, lengths = x
            b = inputs.shape[0]
            inputs = self.embed(inputs)
            x = nn.utils.rnn.pack_padded_sequence(inputs, lengths, batch_first=True, enforce_sorted=False)
        else:
            b = x.shape[0]
            x = self.embed(x)

        h_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)

        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        return self.dropout(final_hidden_state[-1])