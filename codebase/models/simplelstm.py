import torch
import torch.nn as nn
import torch.nn.functional as F
# TODO: add Xavier Initialization


class SimpleLSTM(nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dim, output_dim, dropout=0.3, device=torch.device("cpu")):
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

        self.params = {"vocab": vocab,
                       "embedding_dim": embedding_dim,
                       "hidden_dim": hidden_dim,
                       "output_dim": output_dim,
                       "dropout": dropout,
                       "device": device}

    def forward(self, x, lengths="False"):
        b = x.shape[0]
        x = self.embed(x)
        h_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)
        c_0 = torch.zeros(1, b, self.hidden_dim).to(self.device)
        if lengths != "False":
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, (final_hidden_state, final_cell_state) = self.lstm(x, (h_0, c_0))
        return self.fc_out(self.dropout(final_hidden_state[-1]))
