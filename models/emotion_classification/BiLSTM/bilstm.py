import torch
from torch.nn import functional as F
from torch import nn
torch.manual_seed(200)

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=60, lstm_layer=2, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout = dropout,
                            bidirectional=True)
        self.classify = nn.Linear(hidden_dim*2, 1)

    def forward(self, poses):
        x = torch.transpose(poses, dim0=1, dim1=0)
        lstm_out, (h_n, c_n) = self.lstm(x)
        c_n = c_n.view(self.num_layers, 2, poses.shape[0], self.hidden_dim)
        interim = self.dropout(torch.cat([c_n[-1,i,:,:] for i in range(2)], dim=1))
        out = self.classify(interim)
        out = torch.sigmoid(out)
        return out, torch.Tensor([[0,0,0]])
