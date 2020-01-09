import os
import sys
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from argparse import ArgumentParser

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.data.data_constructors import construct_data_filename


class PoseDataset(Dataset):
	def __init__(self, interval=4, seq_length=4, joint=False, debug=False):
		self.joint = joint
		
		filename = construct_data_filename(interval, seq_length, joint, debug)
		with open(os.path.join(MODELS_DIR, 'data', filename), 'rb') as f:
			self.data = pickle.load(f)
		
		self.actor_pairsA = []
		self.actor_pairsB = []
		self.posesA = []
		self.posesB = []
		self.labelsA =[]
		self.labelsB = []
		self.actorsA = []
		self.actorsB = []
		self.viewsA = []
		self.viewsB = []

		for video_id, views in self.data.items():
			actorA = video_id.split('_')[2][1:]
			actorB = video_id.split('_')[3][1:-4]
			pair = ''.join(sorted([actorA, actorB], key= lambda e: int(e)))
			
			for view, actors in views.items():
				for actor in actors:
					poses = actors[actor]["poses"]
					labels = actors[actor]["labels"]
					if actor == 'A':
						self.actor_pairsA += [pair]*len(poses)
						self.posesA += poses
						self.labelsA += labels
						self.actorsA += [actor]*len(poses)
						self.viewsA += [view]*len(poses)
					else:
						self.actor_pairsB += [pair]*len(poses)
						self.posesB += poses
						self.labelsB += labels
						self.actorsB += [actor]*len(poses)
						self.viewsB += [view]*len(poses)			

		self.unique_actor_pairs = sorted(list(set(self.actor_pairsA.copy())))

		if not joint:
			self.poses = self.posesA + self.posesB
			self.labels = self.labelsA + self.labelsB
			self.actors = self.actorsA + self.actorsB 
			self.actor_pairs = self.actor_pairsA + self.actor_pairsB
			self.views = self.viewsA + self.viewsB
		# 	print("in COnsTRUCTOR")
		# 	print("len actor pairs", len(self.actor_pairs))
		# 	print("len actors", len(self.actors))		
		# 	print("len labels", len(self.labels))
		# 	print("len poses", len(self.poses))
		# 	print("len views", len(self.views))
		# if joint:
		# 	print("in CONSTRUCTOR")
		# 	print("len actor pairs A", len(self.actor_pairsA))
		# 	print("len actor pairs B", len(self.actor_pairsB))
		# 	print("actors A", len(self.actorsA))
		# 	print("actors B", len(self.actorsB))
		# 	print("len labels A", len(self.labelsA))
		# 	print("len labels B", len(self.labelsB))
		# 	print("poses A", len(self.posesA))
		# 	print("poses B", len(self.posesB))
		# 	print("views A", len(self.viewsA))
		# 	print("views B", len(self.viewsB))
	
	def __len__(self):
		if not self.joint:
			return len(self.poses) - 1
		else:
			return len(self.posesA) - 1

	def __getitem__(self, i):
		if not self.joint:
			return {'pose': self.poses[i], 'labels':self.labels[i],
					'actor': self.actors[i], 'actor_pair': self.actor_pairs[i],
					'view':self.views[i]}
		else:
			return {'poseA': self.posesA[i], 'labelsA': self.labelsA[i],
					'poseB': self.posesB[i], 'labelsB': self.labelsB[i],
					'actorA': self.actorsA[i], 'actorB': self.actorsB[i],
					'viewA': self.viewsA[i], 'viewB': self.viewsB[i],
					'actor_pairsA': self.actor_pairsA[i], 'actor_pairsB': self.actor_pairsB[i]}

	def split_data(self, fold):
		dev_pair = self.unique_actor_pairs[fold-1]
		dev_indices = []
		train_indices = []


		if not self.joint:
			for i, actor_pair in enumerate(self.actor_pairs):
				if actor_pair == dev_pair:
					dev_indices.append(i)
				else:
					train_indices.append(i)
		else:
			for i, actor_pair in enumerate(self.actor_pairsA):
				if actor_pair == dev_pair:
					dev_indices.append(i)
				else:
					train_indices.append(i)

		return train_indices, dev_indices



if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=4)
	parser.add_argument('-seq_len', default=4)
	parser.add_argument('-joint', action='store_true', default=False)
	parser.add_argument('-debug', action="store_true", default=False)
	args = parser.parse_args()

	dataset = PoseDataset(args.interval, args.seq_len, args.joint, args.debug)
	
	for item in dataset:
		print(item['labelA'])
		print(item['labelB'])