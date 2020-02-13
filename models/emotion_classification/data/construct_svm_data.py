import os
import sys
import pickle
import numpy as np
from argparse import ArgumentParser
from torch.utils.data import Dataset, Subset, DataLoader
from torch_datasets import PoseDataset
from scipy.signal import argrelextrema
import scipy.stats

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
print(PROJECT_DIR)
from definitions import constants

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
GOLD_STANDARD_PATH = constants["GOLD_STANDARD_PATH"]
MODELS_DIR = constants["MODELS_DIR"]
SVM_PART_PAIRS = constants["SVM_ANGLES"]

"""
this script will construct a pickle file containing all data necessary for all
SVM and Linear models. It computes angles between different body parts
at each timepoint in a pose sequence, before computing statistical properties
of these angles over the entire sequence.

this script cannot be run without many files from previous data preprocessing steps.
these are large and are not included in the repo.
"""


def get_v(body_part1, body_part2):
	"""
	given coordinates for two body parts, calculates the vector between them
	"""
	return np.array([body_part1[0] - body_part2[0],
				body_part1[1] - body_part2[1]])

def get_angle(v1, v2):
	"""
	get the angle between two vectors calculated using function above
	"""
	return np.arccos(np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)))

def keypoint_sequence_to_angles_seq(keypoints_seq):
	"""
	converts a sequence of brute body keypoints to a sequence of body part angles
	"""
	# keypoints_seq = 5 x 50
	angles_seq = []
	keypoints_seq = np.squeeze(keypoints_seq, axis=0)
	for seq in keypoints_seq:
		# unflatten keypoints seq
		seq = seq.reshape(25,2)
		angles = []
		# for each pair of body part vectors, calculate angle between them
		for part_pairs in SVM_PART_PAIRS:

			pair1 = part_pairs[0]
			body_part1 = seq[pair1[0]]
			body_part2 = seq[pair1[1]]
			# if either of the body parts has a zero value, this means it
			# hasn't been properly detected.
			if 0 in body_part1 or 0 in body_part2:
				angles.append(0)
				continue
			# calculate first bodypart vector
			v1 = get_v(body_part1, body_part2)
			pair2 = part_pairs[1]
			body_part1 = seq[pair2[0]]
			body_part2 = seq[pair2[1]]
			if 0 in body_part1 or 0 in body_part2:
				angles.append(0)
				continue
			# calculate second body part vector
			v2 = get_v(body_part1, body_part2)
			# get angle between them and append to angle sequence
			angle = get_angle(v1, v2)
			angles.append(angle)
		angles_seq.append(angles)

	return np.array(angles_seq)

def compute_statistics(angles_seq):
	"""
	Given an angle sequence, will compute a variety of statistics for
	each angle.
	"""
	features = []
	for col_index in range(angles_seq.shape[1]):
		column = angles_seq[:,col_index]
		# num local maxima
		features.append(len(argrelextrema(column, np.greater)[0]))
		features.append(column.mean())
		features.append(column.std())
		features.append(column.max())
		features.append(column.min())
		features.append(scipy.stats.skew(column))
		# num zero crossings
		features.append((np.diff(np.sign(column)) != 0).sum())
		mean_centered = column - column.mean()
		# num mean crossings
		features.append((np.diff(np.sign(mean_centered)) != 0).sum())
	features = np.array(features)

	return features

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_len', default=5, type=int)
	parser.add_argument('-joint', action='store_true', default=True)
	parser.add_argument('-interp', action='store_true', default=True)
	args = parser.parse_args()

	brute_data = PoseDataset(args.interval, args.seq_len, 'all', input='brute', interp=args.interp)

	views = []
	actor_pairs = []
	actors = []
	labels = []
	features = []
	# brute_data = Subset(brute_data, indices=range(200))
	brute_data_loader = DataLoader(brute_data, batch_size=1)
	for datapoint in brute_data_loader:
		views.append(datapoint['view'][0])
		actor_pairs.append(datapoint['actor_pair'][0])
		actors.append(datapoint['actor'])
		labels.append(datapoint['labels'].squeeze(0).tolist())

		poses = datapoint['pose'].numpy()
		angles_seq = keypoint_sequence_to_angles_seq(poses)
		features.append(compute_statistics(angles_seq))

	labels = np.array(labels)
	features = np.array(features)

	data ={'features':features, 'labels':labels, 'actor_pairs':actor_pairs, 'actors':actors, 'views':views}
	print('views', len(views))
	print('actor_pairs', len(actor_pairs))
	print('actors', len(actors))
	print('labels', labels.shape)
	print('features', features.shape)
	if args.interp:
		filename='svm_data_interp.pkl'
	else:
		filename='svm_data.pkl'
	with open(filename, 'wb') as f:
		pickle.dump(data, f)
