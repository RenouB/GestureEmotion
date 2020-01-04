import torch
from torch import nn
from CNN import CNN

class OneActorOneModalityCNN(nn.module):
	def __init__(self, feats_dim, n_filters, filter_sizes, 
				 cnn_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.CNN = CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.dropout = nn.Dropout(dropout)
		self.classify = nn.Linear(cnn_output_dim, num_classes)
	
	def forward(feats):
		feats = self.dropout(feats)
		feats = self.poseCNN(feats)
		feats = self.dropout(feats)
		out = self.classify(feats)

		return out

class TwoActorsOneModalityCNN(nn.module):
	def __init__(self, feats_dim, n_filters, filter_sizes, 
				 cnn_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.CNN = CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.dropout = nn.Dropout(dropout)
		self.classify = nn.Linear(cnn_output_dim * 2, num_classes)
	
	def forward(feats1, feats2):
		feats1 = self.dropout(feats1)
		feats1 = self.CNN(feats1)

		feats2 = self.dropout(feats2)
		feats2 = self.CNN(feats2)
		
		both = torch.cat([feats1, feats2], dim=0)
		both = self.dropout(both)
		out = self.classify(both)

		return out


class OneActorTwoModalities(nn.module):
	def __init__(self, mode1_dim, mode2_dim, n_filters, filter_sizes, 
				 cnn_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.mode1_CNN = CNN(mode1_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.mode2_CNN = CNN(mode2_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.dropout = nn.Dropout(dropout)
		self.classify = nn.Linear(cnn_output_dim * 2, num_classes)
	
	def forward(mode1_feats, mode2_feats):
		mode1_feats = self.dropout(mode1_feats)
		mode1_feats = self.poseCNN(mode1_feats)
		
		mode2_feats = self.dropout(mode2_feats)
		mode2_feats = self.poseCNN(mode2_feats)
		
		out = self.classify(feats)

		return out

class TwoActorsTwoModalities(nn.module):
	def __init__(self, mode1_dim, mode2_dim, n_filters, filter_sizes, 
				 cnn_output_dim, linear_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.mode1_CNN = CNN(mode1_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.mode2_CNN = CNN(mode2_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.actor = nn.Linear(cn_output_dim * 2, linear_output_dim)
		self.dropout = nn.Dropout(dropout)
		
		self.classify = nn.Linear(linear_output_dim * 2, num_classes)
	
	def forward(actor1_feats, actor2_feats):
		actor1_mode1, actor1_mode2 = actor1_feats
		actor2_mode1, actor2_mode2 = actor2_feats

		dropout_mode1 = self.dropout(feats) for feats in [actor1_mode1, actor2_mode1]
		dropout_mode2 = self.dropout(feats) for feats in [actor1_mode2, actor2_mode2]

		modes1 = [mode1_CNN(mode1) for mode1 in dropout_mode1]
		modes2 = [mode2_CNN(mode2) for mode2 in dropout_mode2]

		actor1 = torch.cat([modes1[0], modes2[0]])
		actor1 = self.dropout(actor1)
		actor1 = self.actor(actor1)

		actor2 = torch.cat([modes1[1], modes2[1]])
		actor2 = self.dropout(actor2)
		actor2 = self.actor(actor2)

		both_actors = torch.cat([actor1, actor2], dim=0)
		out = self.classify(both_actors)

		return out