import os
import sys
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
GOLD_STANDARD_PATH = constants["GOLD_STANDARD_PATH"]
MODELS_DIR = constants["MODELS_DIR"]


def construct_data_filename(interval, seq_length, joint, debug):
	if joint:
		joint_str = 'joint-'
	else:
		joint_str = 'ind-'
	if debug:
		debug_str = 'debug-'
	else:
		debug_str = ''
	interval_str = 'int'+str(interval)+'-'
	seq_string = 'seq'+str(seq_length)
	filename = debug_str+joint_str+interval_str+seq_string+'.pkl'

	return filename

def construct_pose_data(interval, seq_length, joint, debug):
	annotations = pd.read_csv(GOLD_STANDARD_PATH)

	if debug:
		with open(os.path.join(PROCESSED_BODY_FEATS_DIR, "debug_cnn.pkl"), "rb") as f:
			data = pickle.load(f)
	else:
		with open(os.path.join(PROCESSED_BODY_FEATS_DIR, "all.pkl"), "rb") as f:
			data = pickle.load(f)

	poses = {'A': [], 'B':[]}
	labels = {'A': [], 'B':[]} 
	print('beginning first iteration')
	for video, views in data.items():
		annotations_video_id = video[:-4]
		for view, actors in views.items():
			for actor, frames in actors.items():
				for frame in range(interval*(seq_length-1), len(frames)):
					sequence_keypoints = [actors[actor][frame_index] for frame_index in 
										range(frame-(interval*(seq_length-1)),frame+1, interval) 
										if type(actors[actor][frame_index]) != str]

					if len(sequence_keypoints) != seq_length:
						poses[actor].append(None)
						labels[actor].append(None)
						continue

					# print(annotations_video_id)
					# print(actor)
					# print(frame)
					current_labels = annotations.loc[(annotations["video_ids"] == annotations_video_id)
					 & (annotations["A_or_B"] == actor) & (annotations["videoTime"] == frame)]
					
					current_labels = current_labels.loc[:,["Anger","Happiness","Sadness","Surprise"]].values[0]
					
					poses[actor].append(sequence_keypoints)
					labels[actor].append(list(current_labels))
		print(video)

	print('beginning second pass')
	filtered_poses = {'A': [], 'B': []}
	filtered_labels = {'A': [], 'B': []}
	if joint:
		for i, pose in enumerate(poses['A']):
			if pose is not None and poses['B'][i] is not None:
				filtered_poses['A'].append(pose)
				filtered_poses['B'].append(poses['B'][i])
				filtered_labels['A'].append(labels['A'][i])
				filtered_labels['B'].append(labels['B'][i])
			else:
				continue
	else:
		filtered_poses['A'] = [pose for pose in poses['A'] if pose is not None]
		filtered_poses['B'] = [pose for pose in poses['B'] if pose is not None]
		filtered_labels['A'] = [label for label in labels['A'] if label is not None]
		filtered_labels['B'] = [label for label in labels['B'] if label is not None]

	print(len(poses['A']), len(poses['B']), len(labels['A']), len(labels['B']))
	print(len(filtered_poses['A']), len(filtered_poses['B']), len(filtered_labels['A']), 
		len(filtered_labels['B']))
	
	filename = construct_data_filename(interval, seq_length, joint, debug)

	with open(os.path.join(MODELS_DIR, 'data', 'poses-'+filename), 'wb') as f:
		pickle.dump(filtered_poses, f)
	with open(os.path.join(MODELS_DIR, 'data', 'labels-'+filename), 'wb') as f:
		pickle.dump(filtered_labels, f)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=4)
	parser.add_argument('-seq_len', default=4)
	parser.add_argument('-joint', action='store_true', default=False)
	parser.add_argument('-debug', action="store_true", default=False)
	args = parser.parse_args()

	construct_pose_data(args.interval, args.seq_len, args.joint, args.debug)