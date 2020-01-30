import sys
import os
import numpy
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_utils import filter_keypoints, get_crop_coordinates, get_body_image_filename, \
convert_keypoints_to_array, crop, add_keypoints_to_sequences, normalize_keypoints, \
interpolate_keypoints_all_frames, interpolate_missing_coordinates

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

MANUALLY_SELECTED_IMAGES_DIR = constants["MANUALLY_SELECTED_IMAGES_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
train = {}
test = {}
all_together ={}
for split, folder in [(train, 'train'), (test, 'test')]:
	for view in os.scandir(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder)):
		for video in os.scandir(view.path):
			print(video.path)
			raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view.name, video.name)

			# put corresponding entry in split
			if video.name not in split:
				split[video.name] = {}
			if video.name not in all_together:
				all_together[video.name] = {}
			if view.name not in split[video.name]:
				split[video.name][view.name] = {"A":{}, "B":{}}
			if view.name not in all_together[video.name]:
				all_together[video.name][view.name] = {"A":{}, "B":{}}

			# begin iterating over json annotations for this file
			video_jsons = sorted(os.listdir(raw_feats_dir))
			for frame_index, js in enumerate(video_jsons):
				with open(os.path.join(raw_feats_dir, js)) as f:
					people = json.load(f)['people']

				all_keypoint_arrays = []
				for person in people:
					keypoints =person['pose_keypoints_2d']
					all_keypoint_arrays.append(convert_keypoints_to_array(keypoints))

				keypoints1, keypoints2 = filter_keypoints(all_keypoint_arrays, view.name)
				keypoints1 = normalize_keypoints(keypoints1)
				keypoints2 = normalize_keypoints(keypoints2)
				actorA_images = sorted(os.listdir(os.path.join(video.path, 'A')))
				actorB_images = sorted(os.listdir(os.path.join(video.path, 'B')))
				if get_body_image_filename(frame_index, 1) in actorA_images or \
					get_body_image_filename(frame_index, 2) in actorB_images:
					assignment = {1:'A', 2:'B'}

				if get_body_image_filename(frame_index, 1) in actorB_images or \
					get_body_image_filename(frame_index, 2) in actorA_images:
					assignment = {1:'B', 2:'A'}

				all_together = add_keypoints_to_sequences(all_together, video.name, view.name,
							 frame_index, assignment, keypoints1, keypoints2)

begin interpolation - replace with nearest neighbour or 2
for video, views in all_together.items():
	for view, actor in views.items():
		for actor, frames in actor.items():
			all_keypoints = np.array([keypoints for keypoints in frames.values()
								if type(keypoints) == np.ndarray])
			


			all_keypoints = interpolate_keypoints_all_frames(all_keypoints)
			for i, keypoints in enumerate(all_keypoints):
				all_together[video][view][actor][i] = keypoints



with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'interp_all_manually_selected_cnn.pkl'), 'wb') as f:
	pickle.dump(all_together, f)
