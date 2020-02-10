import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: add Xavier Initialization
# TODO: set trainable weights to false

class SimpleLSTM(nn.Module):
    def __init__(self, vocab, embedding_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.3, device=torch.device("cpu"),
                 use_lengths=True):
        """

        @param vocab: a vector containing the word embeddings of the word in the train set
        @param embedding_dim: int specifying the dimensionality of the word embeddings
        @param hidden_dim: int specifying number of hidden units in LSTM
        @param output_dim: int specifying the number of output units
        @param dropout: float specifying the dropout rotia
        @param device: torch.device specifying if model is ran on cpu or gpu
        @param use_lengths: boolean specifying whether to remove padding for LSTM input or not
        """
        super(SimpleLSTM, self).__init__()
        self.params = locals()
        self.embed = nn.Embedding(len(vocab), embedding_dim)
        self.embed.weight.data.copy_(vocab.vectors)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.use_lengths = use_lengths
        self.params = {"vocab": vocab,
                       "embedding_dim": embedding_dim,
                       "hidden_dim": hidden_dim,
                       "output_dim": output_dim,
                       "dropout": dropout,
                       "device": device}

    def forward(self, x, lengths=False):
        """
        @param x: tensor of size (batch_size, seq_length)
        @param lengths: if self.use_lengths this is a vector containing the true length of each sequence
        which is then used for removing the padding from the input
        @return: 
        """
        b = x.shape[0]
        x = self.embed(x)
        h_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)
        if self.use_lengths:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        return self.fc_out(self.dropout(final_hidden_state[-1]))
