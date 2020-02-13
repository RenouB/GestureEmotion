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
        self.evidence1 = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        self.evidence2 = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        self.attention_vector = nn.Sequential(nn.Linear(attention_dim, 1), nn.ReLU(), nn.Dropout())
        self.classify = nn.Linear(attention_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, posesA, posesB):

        A = torch.transpose(posesA, dim0=1, dim1=0)
        lstm_outA, (h_nA, c_nA) = self.lstm(A)

        c_nA = c_nA.view(self.num_layers, 2, posesA.shape[0], self.hidden_dim)
        finalA = self.dropout(torch.cat([c_nA[-1,i,:,:] for i in range(2)], dim=1))

        B = torch.transpose(posesB, dim0=1, dim1=0)
        lstm_outB, (h_nB, c_nB) = self.lstm(B)
        c_nB = c_nB.view(self.num_layers, 2, posesB.shape[0], self.hidden_dim)
        finalB = self.dropout(torch.cat([c_nB[-1,i,:,:] for i in range(2)], dim=1))

        evidence1 = self.evidence1(finalA)
        evidence1 = self.evidence2(finalB)
        score1 = self.attention_vector(evidence1)
        score2 = self.attention_vector(evidence2)
        scoresA = self.softmax(torch.cat([score1, score2], dim=0))
        context = evidence1*scores[1] + evidence2*scoresA[2]
        outA = self.classify(context)
        outA = torch.sigmoid(outA)

        evidence1 = self.evidence1(finalB)
        evidence2 = self.evidenceB(finalA)
        score1 = self.attention_vector(evidence1)
        score2 = self.attention_vector(evidence2)
        scoresB = self.softmax(torch.cat([score1, score2], dim=0))
        context = evidence1*scoresB[0] + evidence2*scoreB[1]
        outB = self.classify(context)
        outB = torch.sigmoid(outB)

        return outA, outB, scoresA.detach(), scoresB.detach()
