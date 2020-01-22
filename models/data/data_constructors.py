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
			raw_data = pickle.load(f)
	else:
		with open(os.path.join(PROCESSED_BODY_FEATS_DIR, "all_manually_selected_cnn.pkl"), "rb") as f:
			raw_data = pickle.load(f)

	data = {}
	total_pose_before = 0
	total_pose_after = 0
	total_labels_before = 0
	total_labels_after = 0
	total_deltas_after = 0
	total_delta_deltas_after = 0
	print('beginning first pass')
	for video, views in raw_data.items():
		print(video)
		annotations_video_id = video[:-4]
		if video not in data:
			data[video] ={}
		for view, actors in views.items():
			if view not in data[video]:
				data[video][view] = {}
			for actor, frames in actors.items():
				if actor not in data[video][view]:
					data[video][view][actor] = {'poses':[],'labels':[], 'deltas':[], 'delta_deltas':[]}
				poses = data[video][view][actor]['poses']
				labels = data[video][view][actor]['labels']
				deltas = data[video][view][actor]['deltas']
				delta_deltas = data[video][view][actor]['delta_deltas']
				for frame in range(interval*(seq_length-1), len(frames)):
					sequence_keypoints = [actors[actor][frame_index] for frame_index in
										range(frame-(interval*(seq_length-1)),frame+1, interval)
										if type(actors[actor][frame_index]) != str]

					if len(sequence_keypoints) != seq_length:
						poses.append(None)
						labels.append(None)
						deltas.append(None)
						delta_deltas.append(None)
						continue
					current_labels = annotations.loc[(annotations["video_ids"] \
								== annotations_video_id)
					 & (annotations["A_or_B"] == actor) & (annotations["videoTime"] == frame)]

					if not len(current_labels):
						continue

					current_labels = \
						current_labels.loc[:,["Anger","Happiness","Sadness","Surprise"]].values[0]

					original_poses = [keypoints.flatten() for keypoints in sequence_keypoints]
					perturbed_poses = [np.concatenate([-keypoints[:,0], keypoints[:,1]], axis=0).flatten()
									for keypoints in sequence_keypoints]
					original_deltas = [original_poses[i] - original_poses[i -1] for i in range(1, len(original_poses))]
					perturbed_deltas = [original_poses[i] - original_poses[i -1] for i in range(1, len(original_poses))]

					original_delta_deltas = [original_deltas[i] - original_deltas[i -1] for i in range(1, len(original_deltas))]
					perturbed_deltas_deltas = [original_deltas[i] - original_deltas[i -1] for i in range(1, len(original_deltas))]

					poses.append(original_poses)
					poses.append(perturbed_poses)
					deltas.append(original_deltas)
					deltas.append(perturbed_deltas)
					delta_deltas.append(original_delta_deltas)
					delta_deltas.append(perturbed_deltas_deltas)
					# append original label + copy
					labels.append(np.array(current_labels))
					labels.append(np.array(current_labels))


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
			else:
				filtered['A']['poses'] = [pose for pose in posesA if pose is not None]
				filtered['B']['poses'] = [pose for pose in posesB if pose is not None]
				filtered['A']['labels'] = [label for label in labelsA if label is not None]
				filtered['B']['labels'] = [label for label in labelsB if label is not None]
				filtered['A']['deltas'] = [delta for delta in deltasA if delta is not None]
				filtered['B']['deltas'] = [delta for delta in deltasB if delta is not None]
				filtered['A']['delta_deltass'] = [delta_deltas for delta_deltas in delta_deltassA if delta_deltas is not None]
				filtered['B']['delta_deltass'] = [delta_deltas for delta_deltas in delta_deltassB if delta_deltas is not None]

			total_pose_before += (len(data[video][view]['A']['poses']) \
						+ len(data[video][view]['B']['poses']))
			total_labels_before += len(data[video][view]['A']['labels']) \
						+ len(data[video][view]['B']['labels'])
			total_pose_after += len(filtered['A']['poses']) + len(filtered['B']['poses'])
			total_labels_after += len(filtered['A']['labels']) + len(filtered['B']['labels'])
			total_deltas_after += len(filtered['A']['deltas']) + len(filtered['B']['deltas'])
			total_delta_deltas_after += len(filtered['A']['delta_deltas']) + len(filtered['B']['delta_deltas'])


			data[video][view]['A']['poses'] = filtered['A']['poses']
			data[video][view]['B']['poses'] = filtered['B']['poses']
			data[video][view]['A']['labels'] = filtered['A']['labels']
			data[video][view]['B']['labels'] = filtered['B']['labels']
			data[video][view]['A']['deltas'] = filtered['A']['deltas']
			data[video][view]['B']['deltas'] = filtered['B']['deltas']
			data[video][view]['A']['delta_deltas'] = filtered['A']['delta_deltas']
			data[video][view]['B']['delta_deltas'] = filtered['B']['delta_deltas']

	filename = 'perturb-'+construct_data_filename(interval, seq_length, joint, debug)
	print("Total poses before filtering:", total_pose_before)
	print("Total labels before filtering:", total_labels_before)
	print("Total poses after filtering:", total_pose_after)
	print("Total labels after filtering:", total_labels_after)
	print("Total deltas after filtering:", total_deltas_after)
	print("Total delta_deltas after filtering:", total_delta_deltas_after)

	with open(os.path.join(MODELS_DIR, 'data', filename), 'wb') as f:
		pickle.dump(data, f)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_len', default=5, type=int)
	parser.add_argument('-joint', action='store_true', default=True)
	parser.add_argument('-debug', action="store_true", default=False)
	args = parser.parse_args()

	construct_pose_data(args.interval, args.seq_len, args.joint, args.debug)
