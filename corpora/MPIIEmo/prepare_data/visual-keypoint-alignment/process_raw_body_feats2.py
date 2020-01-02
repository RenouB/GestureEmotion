import sys
import os
import numpy
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from process_utils2 import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images,\
convert_keypoints_to_array, crop_and_assign, add_keypoints_to_sequences, construct_reference_histograms, \
get_all_channels_hist

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
NECK = constants["NECK"]
BODY_CENTER = constants["BODY_CENTER"]

'''
for each view
for each video
extract openpose coords from json
use them to crop images
use color histogram to associate that image to actor A or B
save image to view/video/{A,B}
save dictionary with annotations for all views
'''

# get reference histograms
color = 'hsv'
only_hue = True
distance = 'cor'


reference = construct_reference_histograms(color, only_hue)


all_videos = {}
for view in ["view6"]: #in os.listdir(RAW_BODY_FEATS_DIR):
	view_dir = os.path.join(TEN_FPS_VIEWS_DIR, view)
	for video in [video for video in os.listdir(view_dir) if not video.endswith("images")]:
		cropped_images_dir = os.path.join(TEN_FPS_VIEWS_DIR, view, 'cropped_images', video)
		print(cropped_images_dir)
		os.system("rm -rf {}".format(cropped_images_dir))
		os.system("mkdir -p {}/A".format(cropped_images_dir))
		os.system("mkdir -p {}/B".format(cropped_images_dir))
		
		if video not in all_videos:
			all_videos[video] = {}
			all_videos[video][view] = {"A":{}, "B":{}}

		title_split =  video.split('_')
		actorA = title_split[2][1:3]
		actorB = title_split[3][1:3]
		
		histA = reference[actorA]
		histB = reference[actorB]

		video_dir = os.path.join(view_dir, video)
		raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view, video)
		video_jsons = sorted(os.listdir(raw_feats_dir))

		for frame_index, js in enumerate(video_jsons):
			# print(frame_index)
			with open(os.path.join(raw_feats_dir, js)) as f:
				people = json.load(f)['people']
			
			all_keypoint_arrays = []
			for person in people:
				keypoints =person['pose_keypoints_2d']
				all_keypoint_arrays.append(convert_keypoints_to_array(keypoints))

			keypoints1, keypoints2 = filter_keypoints(all_keypoint_arrays, view)
			
			frame_image = cv2.imread(os.path.join(TEN_FPS_VIEWS_DIR, view, "images", 
									video, get_frame_image_filename(frame_index)))
			try:
				assignment, croppedA, croppedB = crop_and_assign(color, distance, only_hue, keypoints1, keypoints2, 
													frame_image, histA, histB, actorA, actorB)
			except TypeError:
				print('exception')
				continue
			# print("\n")
			# print(type(croppedA))
			# print(type(croppedB))
			write_cropped_images(cropped_images_dir, frame_index, croppedA, croppedB)
			all_videos = add_keypoints_to_sequences(all_videos, video, view, frame_index, 
											assignment, keypoints1, keypoints2)
				
with open(os.path.join(RAW_BODY_FEATS_DIR, 'all_2.pkl'), 'wb') as f:
	pickle.dump(all_videos, f)



