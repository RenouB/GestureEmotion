import sys
import os
import numpy
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_utils import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images,\
convert_keypoints_to_array, crop, assign, add_keypoints_to_sequences, construct_reference_histograms, \
get_all_channels_hist, get_histogram_params

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
NECK = constants["NECK"]
BODY_CENTER = constants["BODY_CENTER"]

reference_one_hists, reference_one_indices = construct_reference_histograms('hsv', True, 32, False)
torso_only = True
all_videos = {}
for view in ['view6']: #os.listdir(RAW_BODY_FEATS_DIR):
	view_dir = os.path.join(TEN_FPS_VIEWS_DIR, view)
	for video in [video for video in os.listdir(view_dir) if not video.endswith("images")]:

		cropped_images_dir = os.path.join(TEN_FPS_VIEWS_DIR, view, 'cropped_images', video)
		video_dir = os.path.join(view_dir, video)
		raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view, video)

		# check if cropped images dir for this video exists. if so remove
		
		# put corresponding entry in all_videos
		if video not in all_videos:
			all_videos[video] = {}
			all_videos[video][view] = {"A":{}, "B":{}}


		title_split =  video.split('_')
		actorA = title_split[2][1:3]
		actorB = title_split[3][1:3]
		

		color, only_hue, num_bins, distance, param_set = get_histogram_params(actorA, actorB)
		
		if color == None:
			continue

		print(cropped_images_dir)
		os.system("rm -rf {}".format(cropped_images_dir))
		os.system("mkdir -p {}/A".format(cropped_images_dir))
		os.system("mkdir -p {}/B".format(cropped_images_dir))
		

		if param_set == 1:
			histA = reference_one_hists[actorA]
			histB = reference_one_hists[actorB]
			comparison_indices = reference_one_indices[actorA]
		# could eventually add more color histogram param sets if if works out
		elif param_set == 2:
			pass

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
			
			frame_image = cv2.imread(os.path.join(TEN_FPS_VIEWS_DIR, view, "images", 
									video, get_frame_image_filename(frame_index)))

			cropped = crop(keypoints1, keypoints2, frame_image, only_torsos=False)
			cropped_torsos = crop(keypoints1, keypoints2, frame_image, only_torsos=True)
			
			if cropped is not None:
			
				if torso_only:
					cropped1, cropped2 = cropped_torsos
					assignment, croppedA, croppedB = assign(cropped1, cropped2, histA, 
													histB, color, distance, only_hue,
													num_bins, comparison_indices, actorA, actorB)
				else:
					cropped1, cropped2 = cropped

					assignment, croppedA, croppedB = assign(cropped1, cropped2, histA, 
													histB, color, distance, only_hue,
													num_bins, comparison_indices, actorA, actorB)
				croppedA = cropped[croppedA]
				croppedB = cropped[croppedB]
				write_cropped_images(cropped_images_dir, frame_index, croppedA, croppedB)
				all_videos = add_keypoints_to_sequences(all_videos, video, view, frame_index, 
												assignment, keypoints1, keypoints2)
				
with open(os.path.join(RAW_BODY_FEATS_DIR, 'all.pkl'), 'wb') as f:
	pickle.dump(all_videos, f)



