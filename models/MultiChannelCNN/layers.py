import torch
from torch.nn import functional as F
from torch import nn

class CNN(nn.Module):
    def __init__(self, pose_dim, n_filters, filter_sizes, output_dim, 
                 dropout):
        
        super().__init__()
        
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1, 
                                              out_channels = n_filters, 
                                              kernel_size = (fs, pose_dim)) 
                                    for fs in filter_sizes
                                    ])
        
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, pose):

        #pose = [batch size, seq len, pose dim]
        pose = pose.unsqueeze(1)
        #pose = [batch size, 1, seq len, pose dim]
        conved = [F.relu(conv(pose)).squeeze(3) for conv in self.convs]
        #conved_n = [batch size, n_filters, seq len - filter_sizes[n] + 1]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]        
        #pooled_n = [batch size, n_filters]
        cat = self.dropout(torch.cat(pooled, dim = 1))
        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)

class Attention(nn.Module):
    def __init__(self, input_dim, att_vector_dim, dropout):

        super().__init__()

        self.linear_blockA = nn.Sequential(nn.Linear(input_dim, att_vector_dim), nn.Dropout(), nn.ReLU())
        self.linear_blockB = nn.Sequential(nn.Linear(input_dim, att_vector_dim), nn.Dropout(), nn.ReLU())
        self.att_vectorA = nn.Linear(att_vector_dim, 1)
        self.att_vectorB = nn.Linear(att_vector_dim, 1)

    def forward(self, feats1, feats2):
        feats11 = self.linear_blockA(feats1)
        feats21 = self.linear_blockB(feats2)
        scoreA = self.att_vectorA(feats11)
        scoreB = self.att_vectorB(feats21)
        scores = torch.cat([scoreA, scoreB], dim=0)
        scores = F.softmax(scores)
        return feats1 * scores[0] + feats2 * scores[1]

class SimpleAttention(nn.Module):
    def __init__(self, input_dim, att_vector_dim, dropout):

        super().__init__()

        self.linear_block = nn.Sequential(nn.Linear(input_dim, att_vector_dim), nn.Dropout(), nn.ReLU())
        self.att_vector = nn.Linear(att_vector_dim, 1)


    def forward(self, feats1, feats2):
        feats11 = self.linear_block(feats1)
        feats21 = self.linear_block(feats2)
        scoreA = self.att_vector(feats11)
        scoreB = self.att_vector(feats21)
        scores = torch.cat([scoreA, scoreB], dim=0)
        scores = F.softmax(scores)
        return feats1 * scores[0] + feats2 * scores[1]




