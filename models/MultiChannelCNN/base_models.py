import torch
from torch.nn import functional as F
from torch import nn
from layers import CNN, Attention, SimpleAttention
torch.manual_seed(200)

class OneActorOneModalityCNN(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, num_classes, dropout):
		
		super().__init__()
		
		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.classify = nn.Linear(project_output_dim, num_classes)
	
		print(self)

	def forward(self, feats):
		feats = self.dropout(feats)

		feats = self.cnn_block(feats)
		feats = self.project(feats)
		feats = self.classify(feats)
		out = torch.sigmoid(feats)
		return out


class TwoActorsOneModalityCNN(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, att_vector_dim, 
				 num_classes, dropout):
		
		super().__init__()
		
		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.attend = Attention(project_output_dim, att_vector_dim, dropout)
		self.classify = nn.Linear(project_output_dim, num_classes)
	
	def forward(self, feats1, feats2):

		feats1 = self.dropout(feats1)
		feats1 = self.cnn_block(feats1)
		feats1 = self.project(feats1)

		feats2 = self.dropout(feats2)
		feats2 = self.cnn_block(feats2)
		feats2 = self.project(feats2)
		
		attend1 = self.attend(feats1, feats2)
		attend1 = self.dropout(attend1)
		attend2 = self.attend(feats2, feats1)
		attend2 = self.dropout(attend2)

		out1 = self.classify(attend1)
		out1 = torch.sigmoid(out1)
		
		out2 = self.classify(attend2)
		out2 = torch.sigmoid(out2)
		
		out = torch.cat([out1, out2], dim=1).reshape((-1,1))
		return out

class TwoActorsOneModalitySimpleCNN(nn.Module):
	def __init__(self, feats_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, att_vector_dim, 
				 num_classes, dropout):
		
		super().__init__()
		
		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(feats_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.project = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.Dropout(), nn.ReLU())
		self.attend = SimpleAttention(project_output_dim, att_vector_dim, dropout)
		self.classify = nn.Linear(project_output_dim, num_classes)
	
	def forward(self, feats1, feats2):

		feats1 = self.dropout(feats1)
		feats1 = self.cnn_block(feats1)
		feats1 = self.project(feats1)

		feats2 = self.dropout(feats2)
		feats2 = self.cnn_block(feats2)
		feats2 = self.project(feats2)
		
		attend1 = self.attend(feats1, feats2)
		attend1 = self.dropout(attend1)
		attend2 = self.attend(feats2, feats1)
		attend2 = self.dropout(attend2)

		out1 = self.classify(attend1)
		out1 = torch.sigmoid(out1)
		
		out2 = self.classify(attend2)
		out2 = torch.sigmoid(out2)
		
		out = torch.cat([out1, out2], dim=1).reshape((-1,1))
		return out

class OneActorTwoModalities(nn.Module):
	def __init__(self, mode1_dim, mode2_dim, n_filters, filter_sizes, 
				 cnn_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.mode1_CNN = CNN(mode1_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.mode2_CNN = CNN(mode2_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.dropout = nn.Dropout(dropout)
		self.classify = nn.Linear(cnn_output_dim * 2, num_classes)
	
	def forward(self, mode1_feats, mode2_feats):
		mode1_feats = self.dropout(mode1_feats)
		mode1_feats = self.poseCNN(mode1_feats)
		
		mode2_feats = self.dropout(mode2_feats)
		mode2_feats = self.poseCNN(mode2_feats)
		
		out = self.classify(feats)

		return out

class TwoActorsTwoModalities(nn.Module):
	def __init__(self, mode1_dim, mode2_dim, n_filters, filter_sizes, 
				 cnn_output_dim, linear_output_dim, num_classes, dropout):
		
		super().__init__()
				
		self.mode1_CNN = CNN(mode1_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.mode2_CNN = CNN(mode2_dim, n_filters, filter_sizes, cnn_output_dim, dropout)
		self.actor = nn.Linear(cn_output_dim * 2, linear_output_dim)
		self.dropout = nn.Dropout(dropout)
		
		self.classify = nn.Linear(linear_output_dim * 2, num_classes)
	
	def forward(self, actor1_feats, actor2_feats):
		actor1_mode1, actor1_mode2 = actor1_feats
		actor2_mode1, actor2_mode2 = actor2_feats

		dropout_mode1 = [self.dropout(feats) for feats in [actor1_mode1, actor2_mode1]]
		dropout_mode2 = [self.dropout(feats) for feats in [actor1_mode2, actor2_mode2]]

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