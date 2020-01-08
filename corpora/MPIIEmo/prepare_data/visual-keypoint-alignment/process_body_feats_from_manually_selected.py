import sys
import os
import numpy
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_utils2 import filter_keypoints, get_crop_coordinates, get_body_image_filename, \
convert_keypoints_to_array, crop, add_keypoints_to_sequences

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

MANUALLY_SELECTED_IMAGES_DIR = constants["MANUALLY_SELECTED_IMAGES_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
all_videos = {}
for view in ['view6']: #os.listdir(RAW_BODY_FEATS_DIR):
	view_dir = os.path.join(MANUALLY_SELECTED_IMAGES_DIR, view)
	for video in os.listdir(view_dir):
		video_dir = os.path.join(view_dir, video)
		raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view, video)
	
		# put corresponding entry in all_videos
		if video not in all_videos:
			all_videos[video] = {}
			all_videos[video][view] = {"A":{}, "B":{}}

		# begin iterating over json annotations for this file
		video_jsons = sorted(os.listdir(raw_feats_dir))
		for frame_index, js in enumerate(video_jsons):
			with open(os.path.join(raw_feats_dir, js)) as f:
				people = json.load(f)['people']
			
			all_keypoint_arrays = []
			for person in people:
				keypoints =person['pose_keypoints_2d']
				all_keypoint_arrays.append(convert_keypoints_to_array(keypoints))

			keypoints1, keypoints2 = filter_keypoints(all_keypoint_arrays, view)
		
			actorA_images = sorted(os.listdir(os.path.join(video_dir, 'A')))
			actorB_images = sorted(os.listdir(os.path.join(video_dir, 'B')))
			print(get_body_image_filename(frame_index, 1))
			print(get_body_image_filename(frame_index, 2))
			print(actorA_images[:20])
			print(actorA_images[:20])
			print(video_dir)
			if get_body_image_filename(frame_index, 1) in actorA_images or \
				get_body_image_filename(frame_index, 2) in actorB_images:
				assignment = {1:'A', 2:'B'}

			if get_body_image_filename(frame_index, 1) in actorB_images or \
				get_body_image_filename(frame_index, 2) in actorA_images:
				assignment = {1:'B', 2:'A'}

			all_videos = add_keypoints_to_sequences(all_videos, video, view, frame_index, 
												assignment, keypoints1, keypoints2)
				
with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'debug_cnn.pkl'), 'wb') as f:
	pickle.dump(all_videos, f)



