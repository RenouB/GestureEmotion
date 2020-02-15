import torch
from torch.nn import functional as F
from torch import nn

"""
A multilayer perception for emotion classification.
"""

class Linear(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden, dropout):
        super().__init__()
        self.start_block = nn.Sequential(nn.Linear(input_dim, output_dim),
                        nn.Dropout(dropout), nn.ReLU())
        self.next_blocks = [nn.Sequential(nn.Linear(output_dim, output_dim),
                        nn.Dropout(dropout), nn.ReLU()) for i in range(num_hidden - 1)]
        self.classify = nn.Linear(output_dim, 1)

    def forward(self, x):
        x = self.start_block(x)
        for block in self.next_blocks:
            x = block(x)
        x = self.classify(x)
        out = torch.sigmoid(x)
        return out, torch.Tensor([[0,0,0]])
