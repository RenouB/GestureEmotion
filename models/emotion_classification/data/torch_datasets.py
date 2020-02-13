import os
import sys
import pickle
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from argparse import ArgumentParser

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
print(PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.emotion_classification.data.data_constructors import construct_data_filename


class SvmPoseDataset(Dataset):
	def __init__(self, emotion=0, interp=True):
		if interp:
			path = constants["INTERP_SVM_DATA_PATH"]
		else:
			path = constants["SVM_DATA_PATH"]
		with open(path, 'rb') as f:
			data = pickle.load(f)
		self.features = data['features']
		self.labels = data['labels'][:,emotion]
		self.actor_pairs = data['actor_pairs']
		self.unique_actor_pairs = constants["ACTOR_PAIRS"]

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, i):
		return {'features': self.features[i], 'label':self.labels[i]}

	def split_data(self, k):
		dev_indices = []
		train_indices = []
		dev_actor_pair = self.unique_actor_pairs[k]
		for i, pair in enumerate(self.actor_pairs):
			if pair == dev_actor_pair:
				dev_indices.append(i)
			else:
				train_indices.append(i)

		return train_indices, dev_indices



class PoseDataset(Dataset):
	def __init__(self, interval=3, seq_length=5, keypoints='full',joint=False,
					emotion=None, input='brute', interp=True):
		self.joint = joint
		if interp:
			path = constants["INTERP_CNN_DATA_PATH"]
		else:
			path = constants["NON_INTERP_CNN_DATA_PATH"]

		with open(path, 'rb') as f:
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
		self.deltasA = []
		self.deltasB = []
		self.delta_deltasA = []
		self.delta_deltasB = []

		for video_id, views in self.data.items():
			actorA = video_id.split('_')[2][1:]
			actorB = video_id.split('_')[3][1:-4]
			pair = ''.join(sorted([actorA, actorB], key= lambda e: int(e)))

			for view, actors in views.items():
				for actor in actors:
					if keypoints == 'all':
						keypoints_to_retain = [i for i in range(25)]
					if keypoints == 'full':
						keypoints_to_retain = constants["WAIST_UP_BODY_PART_INDICES"]
					elif keypoints == "full-hh":
						keypoints_to_retain = constants["FULL-HH"]
					elif keypoints == "full-head":
						keypoints_to_retain =  constants["FULL-HEAD"]
					elif keypoints == "head":
						keypoints_to_retain = constants["HEAD"]
					elif keypoints == "hands":
						keypoints_to_retain = constants["HANDS"]

					poses = np.array(actors[actor]["poses"])
					new_shape = poses.shape[:-1]+(25,2)
					poses = poses.reshape(new_shape)
					poses = poses[:,:,keypoints_to_retain,:]
					poses = poses.reshape(new_shape[:2]+(-1,))
					poses = list(poses)

					deltas = np.array(actors[actor]["deltas"])
					new_shape = deltas.shape[:-1]+(25,2)
					deltas = deltas.reshape(new_shape)
					deltas = deltas[:,:,keypoints_to_retain,:]
					deltas = deltas.reshape(new_shape[:2]+(-1,))
					deltas = list(deltas)

					delta_deltas = np.array(actors[actor]["delta_deltas"])
					new_shape = delta_deltas.shape[:-1]+(25,2)
					delta_deltas = delta_deltas.reshape(new_shape)
					delta_deltas = delta_deltas[:,:,keypoints_to_retain,:]
					delta_deltas = delta_deltas.reshape(new_shape[:2]+(-1,))
					delta_deltas = list(delta_deltas)

					if emotion is not None:
						labels = np.array(actors[actor]["labels"])[:,emotion]
						labels = labels.tolist()

					else:
						labels = actors[actor]["labels"]
					if actor == 'A':
						self.actor_pairsA += [pair]*len(poses)
						self.posesA += poses
						self.deltasA += deltas
						self.delta_deltasA += delta_deltas
						self.labelsA += labels
						self.actorsA += [actor]*len(poses)
						self.viewsA += [view]*len(poses)
					else:
						self.actor_pairsB += [pair]*len(poses)
						self.posesB += poses
						self.deltasB += deltas
						self.delta_deltasB += delta_deltas
						self.labelsB += labels
						self.actorsB += [actor]*len(poses)
						self.viewsB += [view]*len(poses)

		self.unique_actor_pairs = sorted(list(set(self.actor_pairsA.copy())))

		if not joint:
			self.poses = self.posesA + self.posesB
			self.labels = self.labelsA + self.labelsB
			self.deltas = self.deltasA + self.deltasB
			self.delta_deltas = self.delta_deltasA + self.delta_deltasB
			self.actors = self.actorsA + self.actorsB
			self.actor_pairs = self.actor_pairsA + self.actor_pairsB
			self.views = self.viewsA + self.viewsB

	def __len__(self):
		if not self.joint:
			return len(self.poses) - 2
		else:
			return len(self.posesB) - 1

	def __getitem__(self, i):
		if not self.joint:
			return {'pose': self.poses[i], 'labels':self.labels[i],
					'actor': self.actors[i], 'actor_pair': self.actor_pairs[i],
					'view':self.views[i], 'deltas': self.deltas[i],
					'delta_deltas': self.delta_deltas[i]}

		else:
			return {'poseA': self.posesA[i], 'labelsA': self.labelsA[i],
					'poseB': self.posesB[i], 'labelsB': self.labelsB[i],
					'deltasA': self.deltasA[i], 'deltasB':self.deltasB[i],
					'delta_deltasA': self.delta_deltasA[i], 'delta_deltasB':self.delta_deltasB[i],
					'actorA': self.actorsA[i], 'actorB': self.actorsB[i],
					'viewA': self.viewsA[i], 'viewB': self.viewsB[i],
					'actor_pairsA': self.actor_pairsA[i], 'actor_pairsB': self.actor_pairsB[i]}

	def split_data(self, fold, emotion=None):

		dev_pair = self.unique_actor_pairs[fold]
		print("DEV PAIR: ", dev_pair)
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
	args = parser.parse_args()

	dataset = PoseDataset(args.interval, args.seq_len, args.joint)

	for item in dataset:
		print(item['labelA'])
		print(item['labelB'])
