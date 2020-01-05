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
from data_constructors import construct_data_filename
MODELS_DIR = constants["MODELS_DIR"]

class PoseDataset(Dataset):
	def __init__(self, interval=4, seq_length=4, joint=False, debug=False):
		self.joint = joint
		
		filename = construct_data_filename(interval, seq_length, joint, debug)
		with open(os.path.join(MODELS_DIR, 'data', 'labels-'+filename), 'rb') as f:
			self.labels = pickle.load(f)
			
		with open(os.path.join(MODELS_DIR, 'data', 'poses-'+filename), 'rb') as f:
			self.poses = pickle.load(f)
	
		if not joint:
			self.poses = self.poses['A'] + self.poses['B']
			self.labels = self.labels['A'] + self.labels['B']

	def __len__(self):
		if not self.joint:
			return len(self.poses)
		else:
			return len(self.poses['A'])

	def __getitem__(self, i):
		if not self.joint:
			return {'pose':self.poses[i], 'labels':self.labels[i]}
		else:
			return {'poseA': self.poses['A'][i], 'labelA': self.labels['A'][i],
				'poseB': self.poses['B'][i], 'labelB': self.labels['B'][i]}


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