import os
import sys
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from torch.utils.data import Dataset, Subset

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
GOLD_STANDARD_PATH = constants["GOLD_STANDARD_PATH"]
MODELS_DIR = constants["MODELS_DIR"]

"""
this script will construct a pickle file containing all data necessary for all
brute, delta delta models of the emotion classification task

this script cannot be run without many files from previous data preprocessing steps.
these are large and are not included in the repo.
"""

def construct_data_filename(interval, seq_length, joint):
	"""
	given certain parametres, construct an appropriate filename for pickled
	data file
	"""
	if joint:
		joint_str = 'joint-'
	else:
		joint_str = 'ind-'
	interval_str = 'int'+str(interval)+'-'
	seq_string = 'seq'+str(seq_length)
	filename = joint_str+interval_str+seq_string+'.pkl'
	return filename

def construct_pose_data(interval, seq_length, joint, debug, interp):
	"""
	construct a dictionary of pose data in the following form:
	{video_id: {view: {actor: {poses, labels, deltas, delta_deltas}}}

	each pose in poses is a sequence of poses.

	interval: time separating poses in sequence.
	seq_length: length of sequence in number of poses
	joint: whether or not to only use frames where poses are present for both
	actors
	interp: whether or not to use interpolated data
	"""

	# load gold standard annotations
	annotations = pd.read_csv(GOLD_STANDARD_PATH)

	# load raw keypoints
	if not args.interp:
		with open(os.path.join(PROCESSED_BODY_FEATS_DIR, "all_manually_selected_cnn.pkl"), "rb") as f:
			raw_data = pickle.load(f)
	else:
		with open(os.path.join(PROCESSED_BODY_FEATS_DIR, "interp_all_manually_selected_cnn.pkl"), "rb") as f:
			raw_data = pickle.load(f)
	data = {}
	# these variables are just to count some intersting things
	total_pose_before = 0
	total_pose_after = 0
	total_labels_before = 0
	total_labels_after = 0
	total_deltas_after = 0
	total_delta_deltas_after = 0
	print('beginning first pass')
	# iterate through videos and views of that video
	for video, views in raw_data.items():
		# print video to keep track of status
		print(video)
		# cut off the end of video name to get it's ID in the annotations
		annotations_video_id = video[:-4]
		if video not in data:
			data[video] ={}
		# begin iterating over actors A and B in a given view
		for view, actors in views.items():
			if view not in data[video]:
				data[video][view] = {}
			# begin iterating over all frames for a given actor
			for actor, frames in actors.items():
				if actor not in data[video][view]:
					data[video][view][actor] = {'poses':[],'labels':[],
										'deltas':[], 'delta_deltas':[]}
				poses = data[video][view][actor]['poses']
				labels = data[video][view][actor]['labels']
				deltas = data[video][view][actor]['deltas']
				delta_deltas = data[video][view][actor]['delta_deltas']

				# the heart of data construction begins here.
				# iterate over all available frames of a video in order to
				# construct sequences
				for frame in range(interval*(seq_length-1), len(frames)):
					# if a sequence is string type, this means it's been filtered
					# in a previous step. do not accept it in the sequence; it's
					# not intact.
					sequence_keypoints = [actors[actor][frame_index] for frame_index in
										range(frame-(interval*(seq_length-1)),frame+1, interval)
										if type(actors[actor][frame_index]) != str]

					# if this sequence isn't the right length, append none to all
					# relevant data structures and break loop
					if len(sequence_keypoints) != seq_length:
						poses.append(None)
						labels.append(None)
						deltas.append(None)
						delta_deltas.append(None)
						continue

					# if the sequence has passed the test, fetch the corresponding
					# annotations
					current_labels = annotations.loc[(annotations["video_ids"] \
								== annotations_video_id)
					 & (annotations["A_or_B"] == actor) & (annotations["videoTime"] == frame)]

					# sometimes it is possible that a properly captured sequence
					# is not actually present in the annotations.
					if not len(current_labels):
						continue

					current_labels = \
						current_labels.loc[:,["Anger","Happiness","Sadness","Surprise"]].values[0]

					# flatten the poses
					original_poses = [keypoints.flatten() for keypoints in sequence_keypoints]
					# invert poses around the y axis to perturb
					# then begin constructing deltas and delta deltas
					perturbed_poses = [np.concatenate([-keypoints[:,0], keypoints[:,1]], axis=0).flatten()
									for keypoints in sequence_keypoints]
					original_deltas = [original_poses[i] - original_poses[i -1]
										for i in range(1, len(original_poses))]
					perturbed_deltas = [perturbed_poses[i] - perturbed_poses[i -1]
									for i in range(1, len(original_poses))]

					original_delta_deltas = [original_deltas[i] - original_deltas[i -1]
									for i in range(1, len(original_deltas))]
					perturbed_deltas_deltas = [perturbed_deltas[i] - perturbed_deltas[i -1]
									for i in range(1, len(original_deltas))]

					# append all the things we've just created to the relevant
					# lists in our big dictionary.
					poses.append(original_poses)
					poses.append(perturbed_poses)
					deltas.append(original_deltas)
					deltas.append(perturbed_deltas)
					delta_deltas.append(original_delta_deltas)
					delta_deltas.append(perturbed_deltas_deltas)
					# append original label + copy
					labels.append(np.array(current_labels))
					labels.append(np.array(current_labels))

			# now that we've been through everything once, let's go back through and filter.
			filtered = {'A': {'poses':[], 'labels':[], 'deltas':[], 'delta_deltas':[]},
						'B': {'poses':[],'labels':[], 'deltas':[], 'delta_deltas':[]}}
			posesA = data[video][view]['A']['poses']
			posesB = data[video][view]['B']['poses']
			labelsA = data[video][view]['A']['labels']
			labelsB = data[video][view]['B']['labels']
			deltasA = data[video][view]['A']['deltas']
			deltasB = data[video][view]['B']['deltas']
			delta_deltasA = data[video][view]['A']['delta_deltas']
			delta_deltasB = data[video][view]['B']['delta_deltas']
			if joint:
				for i in range(min(len(posesA), len(posesB))):
					# if in joint setting,
					# only allow datapoints into filtered data structure
					# if information for both actors at that timepoint is present.
					if posesA[i] is not None and posesB[i] is not None:
						filtered['A']['poses'].append(posesA[i])
						filtered['B']['poses'].append(posesB[i])
						filtered['A']['labels'].append(labelsA[i])
						filtered['B']['labels'].append(labelsB[i])
						filtered['A']['deltas'].append(deltasA[i])
						filtered['B']['deltas'].append(deltasB[i])
						filtered['A']['delta_deltas'].append(delta_deltasA[i])
						filtered['B']['delta_deltas'].append(delta_deltasB[i])

					else:
						continue
			# if not in joint setting, then only filter when information
			# is missing for a given actor.
			else:
				filtered['A']['poses'] = [pose for pose in posesA if pose is not None]
				filtered['B']['poses'] = [pose for pose in posesB if pose is not None]
				filtered['A']['labels'] = [label for label in labelsA if label is not None]
				filtered['B']['labels'] = [label for label in labelsB if label is not None]
				filtered['A']['deltas'] = [delta for delta in deltasA if delta is not None]
				filtered['B']['deltas'] = [delta for delta in deltasB if delta is not None]
				filtered['A']['delta_deltas'] = [delta_deltas for delta_deltas in delta_deltassA if delta_deltas is not None]
				filtered['B']['delta_deltas'] = [delta_deltas for delta_deltas in delta_deltassB if delta_deltas is not None]

			# update the things we're counting
			total_pose_before += (len(data[video][view]['A']['poses']) \
						+ len(data[video][view]['B']['poses']))
			total_labels_before += len(data[video][view]['A']['labels']) \
						+ len(data[video][view]['B']['labels'])
			total_pose_after += len(filtered['A']['poses']) + len(filtered['B']['poses'])
			total_labels_after += len(filtered['A']['labels']) + len(filtered['B']['labels'])
			total_deltas_after += len(filtered['A']['deltas']) + len(filtered['B']['deltas'])
			total_delta_deltas_after += len(filtered['A']['delta_deltas']) + len(filtered['B']['delta_deltas'])

			# replace information in the data dictionary by the new filtered
			# information
			data[video][view]['A']['poses'] = filtered['A']['poses']
			data[video][view]['B']['poses'] = filtered['B']['poses']
			data[video][view]['A']['labels'] = filtered['A']['labels']
			data[video][view]['B']['labels'] = filtered['B']['labels']
			data[video][view]['A']['deltas'] = filtered['A']['deltas']
			data[video][view]['B']['deltas'] = filtered['B']['deltas']
			data[video][view]['A']['delta_deltas'] = filtered['A']['delta_deltas']
			data[video][view]['B']['delta_deltas'] = filtered['B']['delta_deltas']

	# construct appropriate filename
	if args.interp:
		filename = 'interp-perturb-'+construct_data_filename(interval, seq_length, joint)
	else:
		filename = 'perturb-'+construct_data_filename(interval, seq_length, joint)

	# save
	with open(os.path.join(MODELS_DIR, 'emotion_classification', 'data', filename), 'wb') as f:
		pickle.dump(data, f)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_len', default=5, type=int)
	parser.add_argument('-joint', action='store_true', default=True)
	parser.add_argument('-interp', action='store_true', default=True)
	args = parser.parse_args()

	construct_pose_data(args.interval, args.seq_len, args.joint, False, args.interp)
