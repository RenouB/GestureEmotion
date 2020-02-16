import torch
from torch.nn import functional as F
from torch import nn

"""
Contains a module for a CNN block, a CNN based architecture
that takes brute keypoint features as input,
and an AttCNN architecture that takes delta delta features as input
"""

class CNN(nn.Module):
	def __init__(self, input_dim, n_filters, filter_sizes, output_dim,
				 dropout):
		"""
		input_dim: dimensions of body keypoint input
		n_filters: num convolutional filters of each size
		filter sizes: a list  indicating desired filters
		output dim: concatenanted output of filters is projected
		to this dimension
		"""
		super().__init__()
		# initialize convolutional layers
		self.convs = nn.ModuleList([
									nn.Conv2d(in_channels = 1,
											  out_channels = n_filters,
											  kernel_size = (fs, input_dim))
									for fs in filter_sizes
									])
		# initialize projection layer
		self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, pose):

		pose = pose.unsqueeze(1)
		# input pose to all convolutional blocks
		conved = [F.relu(conv(pose)).squeeze(3) for conv in self.convs]
		# max pooling
		pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
		# concatenate output and pass to feed forward layer
		cat = self.dropout(torch.cat(pooled, dim = 1))
		return self.fc(cat)

class OneActorOneModalityBrute(nn.Module):
	def __init__(self, input_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, num_classes, dropout):
		"""
		classifies emotions for one actor at a time using brute
		keypoint features.

		input_dim: dimensionality of pose input
		n_filters: num filters of each size in convolutional block
		filter_sizes: list of filter sizes
		project_output_dim: dimensionality of projection just before
		classification
		"""
		super().__init__()

		self.dropout = nn.Dropout(dropout)
		self.cnn_block = nn.Sequential(CNN(input_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
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
		return out, torch.Tensor([[0,0,0]])

class OneActorOneModalityDeltas(nn.Module):
	def __init__(self, input_dim, n_filters, filter_sizes,
				 cnn_output_dim, project_output_dim, attention_dim,
				 num_classes, dropout):
		"""
		classifies emotions for one actor at a time using delta
		keypoint features.

		separate convolutional blocks create representations for
		brute, delta and delta delta features.

		each channel has its own projection matrix which
		outputs an evidence vector.

		a single attention vector generates scores for each evidence,
		which are normalized using the softmax function and subsequently
		used to reconstruct a context vector.

		input_dim: dimensionality of pose input
		n_filters: num filters of each size in convolutional block
		filter_sizes: list of filter sizes
		project_output_dim: dimensionality of projection just before
		classification
		"""

		super().__init__()

		self.dropout = nn.Dropout(dropout)
		# initialize one cnn block for each filter type
		self.cnn_block1 = nn.Sequential(CNN(input_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
										nn.ReLU())
		self.cnn_block2 = nn.Sequential(CNN(input_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
								nn.ReLU())
		self.cnn_block3 = nn.Sequential(CNN(input_dim, n_filters, filter_sizes, cnn_output_dim, dropout),
						nn.ReLU())
		# feed forward layers will project output of cnn blocks
		self.project1 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		self.project2 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		self.project3 = nn.Sequential(nn.Linear(cnn_output_dim, project_output_dim), nn.ReLU(), nn.Dropout())
		# feed forward layers will transform previous projections into evidence
		self.evidence1 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		self.evidence2 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		self.evidence3 = nn.Sequential(nn.Linear(project_output_dim, attention_dim), nn.ReLU(), nn.Dropout())
		# evidence will be scored and normalized using attention and softmax
		self.attention_vector = nn.Sequential(nn.Linear(attention_dim, 1), nn.ReLU(), nn.Dropout())
		self.softmax = nn.Softmax(dim=1)
		self.classify = nn.Linear(project_output_dim, num_classes)

	def forward(self, poses, deltas, delta_deltas):
		poses, deltas, delta_deltas = [self.dropout(input) for input in
										[poses, deltas, delta_deltas]]

		poses = self.cnn_block1(poses)
		poses = self.project1(poses)
		evidence1 = self.evidence1(poses)
		score1 = self.attention_vector(evidence1)

		deltas = self.cnn_block2(deltas)
		deltas = self.project2(deltas)
		evidence2 = self.evidence2(deltas)
		score2 = self.attention_vector(evidence2)

		delta_deltas = self.cnn_block3(delta_deltas)
		delta_deltas = self.project3(delta_deltas)
		evidence3 = self.evidence2(delta_deltas)
		score3 = self.attention_vector(evidence3)
		scores = self.softmax(torch.cat([score1, score2, score3],dim=1))

		context = scores[:,0,None]*poses + scores[:,1,None]*deltas + scores[:,2,None]*delta_deltas
		context = self.classify(context)
		out = torch.sigmoid(context)
		return out, scores.detach()
