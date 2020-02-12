import torch
from torch.nn import functional as F
from torch import nn
torch.manual_seed(200)

class JointBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=60, attention_dim=60, lstm_layer=2, dropout=0.5):
        super(JointBiLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout = dropout,
                            bidirectional=True)
        self.evidenceA = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        self.evidenceB = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        self.attention_vector = nn.Sequential(nn.Linear(attention_dim, 1), nn.ReLU(), nn.Dropout())
        self.classify = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=0)
        # print(self)
    def forward(self, posesA, posesB):

        A = torch.transpose(posesA, dim0=1, dim1=0)
        lstm_outA, (h_nA, c_nA) = self.lstm(A)

        c_nA = c_nA.view(self.num_layers, 2, posesA.shape[0], self.hidden_dim)
        finalA = self.dropout(torch.cat([c_nA[-1,i,:,:] for i in range(2)], dim=1))

        B = torch.transpose(posesB, dim0=1, dim1=0)
        lstm_outB, (h_nB, c_nB) = self.lstm(B)
        c_nB = c_nB.view(self.num_layers, 2, posesB.shape[0], self.hidden_dim)
        finalB = self.dropout(torch.cat([c_nB[-1,i,:,:] for i in range(2)], dim=1))

        evidenceA = self.evidenceA(finalA)
        evidenceB = self.evidenceB(finalB)
        scoreA = self.attention_vector(evidenceA)
        scoreB = self.attention_vector(evidenceB)
        scoresA = self.softmax(torch.cat([scoreA, scoreB], dim=0))
        att_weightsA = torch.cat([scoreA, scoreB], dim=0)
        context = evidenceA*scoresA[0] + evidenceB*scoresA[1]
        outA = self.classify(context)
        outA = torch.sigmoid(outA)

        evidenceB = self.evidenceA(finalB)
        evidenceA = self.evidenceB(finalA)
        scoreA = self.attention_vector(evidenceA)
        scoreB = self.attention_vector(evidenceB)
        scoresB = self.softmax(torch.cat([scoreA, scoreB], dim=0))
        context = evidenceA*scoresB[0] + evidenceB*scoreB[1]
        outB = self.classify(context)
        outB = torch.sigmoid(outB)

        return outA, outB, scoresA.detach(), scoresB.detach()
