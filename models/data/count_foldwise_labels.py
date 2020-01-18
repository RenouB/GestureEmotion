import os
import sys
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset, Subset

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

from torch_datasets import PoseDataset


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=3)
	parser.add_argument('-seq_len', default=5)
	parser.add_argument('-joint', action='store_true', default=False)
	parser.add_argument('-debug', action="store_true", default=False)
	args = parser.parse_args()

	dataset = PoseDataset(args.interval, args.seq_len, args.joint, args.debug)
	print(dataset.unique_actor_pairs)
	for k in range(8):
		train_indices, dev_indices = dataset.split_data(k)
		dev_data = Subset(dataset, dev_indices)
		all_labels = []
		for datapoint in dev_data:
			all_labels.append(datapoint['labels'])
		all_labels = np.array(all_labels)
		all_labels = np.sum(all_labels, axis=0)
		print('\n')
		print(dataset.unique_actor_pairs[k])
		print(all_labels)
		print(len(dev_indices))
