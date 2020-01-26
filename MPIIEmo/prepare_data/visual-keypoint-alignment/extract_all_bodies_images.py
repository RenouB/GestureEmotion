import sys
import os
import json
import numpy as np
import cv2
from process_utils2 import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images_actor_irrelevant,\
crop, convert_keypoints_to_array

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
MANUALLY_SELETED_IMAGES_DIR = constants["MANUALLY_SELECTED_IMAGES_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
NECK = constants["NECK"]
BODY_CENTER = constants["BODY_CENTER"]

for view in ['view4']: #os.listdir(RAW_BODY_FEATS_DIR):
	view_dir = os.path.join(TEN_FPS_VIEWS_DIR, view)
	for video in [video for video in os.listdir(view_dir) if not video.endswith("images")]:

		cropped_images_dir = os.path.join(MANUALLY_SELETED_IMAGES_DIR, view, video)
		video_dir = os.path.join(view_dir, video)
		raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view, video)

		# check if cropped images dir for this video exists. if so remove
		
		
		print(cropped_images_dir)
		os.system("rm -rf {}".format(cropped_images_dir))
		os.system("mkdir -p {}/A".format(cropped_images_dir))
		os.system("mkdir -p {}/B".format(cropped_images_dir))
		

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
			
			if cropped is not None:
				
				cropped1, cropped2 = cropped

				write_cropped_images_actor_irrelevant(cropped_images_dir, frame_index, cropped1, cropped2)
				


