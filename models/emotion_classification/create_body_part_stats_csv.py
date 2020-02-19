import pickle
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.emotion_classification.data.datasets import PoseDataset
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import scipy.stats as stats
from scipy.signal import argrelextrema


"""
so wtf do I even want with this script?
I wanna look at how different body part keypoints correlate with different emotions
24 different keypoints
but each has x and y coordinates
compute stats over x, y coords of each keypoint
get correlations of each of these wrt to labels
aggregate correlations depending on body part groups
do this for each emotion

total missing keypoints -
body part - count
body sections - count


EMOTION DF
body part - x - stat - value -  0c 1c 2c 3c
body_part - y - stat - value - 0c 1c 2c 3c
do this for all body parts
then at the end group them together and average for body part subsets
"""

full_head = constants["FULL-HEAD"]
full_hh = constants["FULL-HH"]
head = constants["HEAD"]
hands = constants["HANDS"]
index_to_body_part = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:15, 9:16, 10:17, 11:18}

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-emotion', default=0)
	parser.add_argument('-keypoints', default="full")
	args = parser.parse_args()

	labels = []
	for emotion in [0,1,2,3]:
		data = PoseDataset(3, 5, args.keypoints, False, emotion, 'brute', True)
		data_loader = DataLoader(data, batch_size=len(data))
		data = next(iter(data_loader))
		labels.append(data['labels'].numpy())
	poses = data["pose"]
	# convert poses to batch * sequence length * 12 * 2
	# missing_keypoint_counts = {i:0 for i in index_to_body_part.values()}
	dict_for_df = {"datapoint":[], "body_part":[], "dim":[], "mean":[],"min":[], "max":[],"variance":[],
	"kurtosis":[],"skewness":[], "rel_max":[], "displacement":[], "anger":[], "happiness":[],
	"sadness":[], "surprise":[]}
	poses = poses.reshape(poses.shape[0], poses.shape[1], 12, 2)
	# apply describe to x coords of every sequence
	for data_index, datapoint in enumerate(poses):
		if data_index % 100 == 0:
			print(data_index)
		x_coords = datapoint[:,:,0]
		y_coords = datapoint[:,:,1]
		# missing_keypoints = (y_coords + x_coords == 0)
		# total_missing_keypoints = missing_keypoints.sum(axis=0)
		# for i in range(12):
		# 	missing_keypoint_counts[index_to_body_part[i]] += total_missing_keypoints[i]
		# missing_keypoint_counts[i] +=
		x_desc = stats.describe(x_coords,axis=0)
		y_desc = stats.describe(y_coords,axis=0)
		x_rel_max = np.apply_along_axis(lambda x: len(argrelextrema(x, np.greater)[0]),
					axis=0, arr=x_coords)
		y_rel_max = np.apply_along_axis(lambda x: len(argrelextrema(x, np.greater)[0]),
					axis=0, arr=y_coords)
		# mean_centered_x = x_coords - np.outer(x_means, np.ones(12))
		x_displacement = np.abs(np.diff(x_coords, axis=0)).sum(axis=0)
		# mean_centered_y = y_coords - np.outer(y_means, np.ones(12))
		y_displacement = np.abs(np.diff(y_coords, axis=0)).sum(axis=0)


		for ii, desc in enumerate([x_desc, y_desc]):
			dict_for_df["body_part"] += [index_to_body_part[i] for i in range(12)]
			dict_for_df["datapoint"] += [data_index for i in range(12)]
			dict_for_df["anger"] += [labels[0][data_index] for i in range(12)]
			dict_for_df["happiness"] += [labels[1][data_index] for i in range(12)]
			dict_for_df["sadness"] += [labels[2][data_index] for i in range(12)]
			dict_for_df["surprise"] += [labels[3][data_index] for i in range(12)]
			if ii == 0:
				dict_for_df["dim"] += ["x" for i in range(12)]
				dict_for_df["rel_max"] += x_rel_max.tolist()
				dict_for_df["displacement"] += x_displacement.tolist()
			else:
				dict_for_df["dim"] += ["y" for i in range(12)]
				dict_for_df["rel_max"] += y_rel_max.tolist()
				dict_for_df["displacement"] += y_displacement.tolist()
			# dict_for_df["missing"] += total_missing_keypoints.tolist()
			dict_for_df["mean"] += desc.mean.tolist()
			dict_for_df["variance"] += desc.variance.tolist()
			dict_for_df["skewness"] += desc.skewness.tolist()
			dict_for_df["kurtosis"] += desc.kurtosis.tolist()
			dict_for_df["min"] += desc.minmax[0].tolist()
			dict_for_df["max"] += desc.minmax[1].tolist()




	for key, value in dict_for_df.items():
		print(key, len(value))
	df = pd.DataFrame(dict_for_df)
	df.to_csv("body_keypoint_stats.csv")
# print(missing_keypoint_counts)
