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
        self.linear_block = nn.Sequential(nn.Linear(input_dim, att_vector_dim), nn.Dropout(), nn.ReLU())
        self.att_vector = nn.Linear(att_vector_dim, 1)

    def forward(self, feats):
        projection = self.linear_block(feats)
        score = self.att_vector(projection)
        return score
