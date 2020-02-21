import torch
import torch.nn as nn
# TODO: add Xavier Initialization
# TODO: set trainable weights to false


class SimpleLSTM(nn.Module):
    def __init__(self, vocab, hidden_dim: int, output_dim: int, dropout: float = 0.3,
                 device=torch.device("cpu"), use_lengths=True):
        """
        @param vocab: a vector containing the word embeddings of the word in the train set
        @param embedding_dim: int specifying the dimensionality of the word embeddings
        @param hidden_dim: int specifying number of hidden units in LSTM
        @param output_dim: int specifying the number of output units
        @param dropout: float specifying the dropout ratio
        @param device: torch.device specifying if model is ran on cpu or gpu
        @param use_lengths: boolean specifying whether to remove padding for LSTM input or not
        """
        super(SimpleLSTM, self).__init__()
        self.params = locals()
        self.embed = nn.Embedding(*vocab.shape)
        self.embed.weight.data.copy_(vocab)
        self.embedding_dim = vocab.shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.use_lengths = use_lengths
        self.params = {"vocab": vocab,
                       "embedding_dim": self.embedding_dim,
                       "hidden_dim": hidden_dim,
                       "output_dim": output_dim,
                       "dropout": dropout,
                       "device": device}

    def forward(self, x):
        """
        @param x: tensor of size (batch_size, seq_length)
        @return: 
        """
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
        return self.fc_out(self.dropout(final_hidden_state[-1]))
