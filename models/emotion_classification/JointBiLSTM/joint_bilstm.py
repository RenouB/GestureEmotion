import torch
from torch.nn import functional as F
from torch import nn

"""
JointBiLSTM model. This model uses an attention mechanism to adaptively
weight representations of actors A and B to reconstruct a context vector
which is used to classify.
""""

class JointBiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=60, attention_dim=60, lstm_layer=2, dropout=0.5):
        super(JointBiLSTM, self).__init__()
        # dimensionality of input features
        self.input_dim = input_dim
        # dimensionality of BiLSTM hidden state output
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = lstm_layer
        self.lstm = nn.LSTM(input_size=self.input_dim,
                            hidden_size=hidden_dim,
                            num_layers=lstm_layer,
                            dropout = dropout,
                            bidirectional=True)
        # evidence matrices project hidden state outputs for actors 1 and 2
        self.evidence1 = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        self.evidence2 = nn.Sequential(nn.Linear(hidden_dim*2, attention_dim), nn.ReLU(), nn.Dropout())
        # evidence vector uses evidences 1 and 2 to generate score
        self.attention_vector = nn.Sequential(nn.Linear(attention_dim, 1), nn.ReLU(), nn.Dropout())
        # softmax is used to normalize attention scores
        self.softmax = nn.Softmax(dim=1)
        self.classify = nn.Linear(attention_dim, 1)

    def forward(self, posesA, posesB):

        # feed poseA to BiLSTM
        A = torch.transpose(posesA, dim0=1, dim1=0)
        lstm_outA, (h_nA, c_nA) = self.lstm(A)

        # retrieve final hidden state from final layer
        c_nA = c_nA.view(self.num_layers, 2, posesA.shape[0], self.hidden_dim)
        finalA = self.dropout(torch.cat([c_nA[-1,i,:,:] for i in range(2)], dim=1))

        B = torch.transpose(posesB, dim0=1, dim1=0)
        lstm_outB, (h_nB, c_nB) = self.lstm(B)
        c_nB = c_nB.view(self.num_layers, 2, posesB.shape[0], self.hidden_dim)
        finalB = self.dropout(torch.cat([c_nB[-1,i,:,:] for i in range(2)], dim=1))

        # classify actor A. in this scenario, A is fed to evidence 1
        # and B to evidence 2
        evidence1 = self.evidence1(finalA)
        evidence2 = self.evidence2(finalB)
        score1 = self.attention_vector(evidence1)
        score2 = self.attention_vector(evidence2)
        scoresA = self.softmax(torch.cat([score1, score2], dim=0))
        context = evidence1*scores[1] + evidence2*scoresA[2]
        outA = self.classify(context)
        outA = torch.sigmoid(outA)

        # classify actor B. The opposite occurs; B is fed to evidence 1
        # and A to evidence 2
        evidence1 = self.evidence1(finalB)
        evidence2 = self.evidenceB(finalA)
        score1 = self.attention_vector(evidence1)
        score2 = self.attention_vector(evidence2)
        scoresB = self.softmax(torch.cat([score1, score2], dim=0))
        context = evidence1*scoresB[0] + evidence2*scoreB[1]
        outB = self.classify(context)
        outB = torch.sigmoid(outB)

        # return outputs and attention scores
        return outA, outB, scoresA.detach(), scoresB.detach()
