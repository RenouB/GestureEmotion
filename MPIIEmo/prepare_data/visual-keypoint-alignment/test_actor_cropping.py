import sys
import os
import numpy
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from process_utils2 import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images, \
convert_keypoints_to_array, crop_and_assign, add_keypoints_to_sequences, construct_reference_histograms

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

ACTOR_REFERENCE_IMAGES_DIR = constants["ACTOR_REFERENCE_IMAGES_DIR"]
TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
NECK = constants["NECK"]
BODY_CENTER = constants["BODY_CENTER"]

''' what do I want to be able to do with this? 
for a frame I wanna crop to keypoints and display'''




if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-frame", default=0, type=int)
	parser.add_argument("-video", type=str)
	parser.add_argument("-view", type=int)
	parser.add_argument("-color", default="rgb")
	parser.add_argument("-distance", default="cor", )
	args = parser.parse_args()

	view = "view"+str(args.view)
	os.system("rm -rf tmp && mkdir tmp")
	os.system("mkdir -p tmp/A")
	os.system("mkdir -p tmp/B")
		
	reference = construct_reference_histograms(args.color)
		
	title_split =  args.video.split('_')
	actorA = title_split[2][1:3]
	actorB = title_split[3][1:3]
	print("actorA:", actorA, "actorB", actorB)

	histA = reference[actorA]
	histB = reference[actorB]

	raw_feats_dir = os.path.join(RAW_BODY_FEATS_DIR, view, args.video)
	
	if args.frame:
		video_jsons = ['_'.join([args.video[:-4], '0'*(12-len(str(args.frame)))+str(args.frame), 'keypoints.json'])]
	else:
		video_jsons = sorted(os.listdir(raw_feats_dir))



	for frame_index, js in enumerate(video_jsons):
		with open(os.path.join(raw_feats_dir, js)) as f:
			people = json.load(f)['people']

		all_keypoint_arrays = []
		for person in people:
			keypoints =person['pose_keypoints_2d']
			all_keypoint_arrays.append(convert_keypoints_to_array(keypoints))

		keypoints1, keypoints2 = filter_keypoints(all_keypoint_arrays, view)
		# print("FILTERED KEYPOINTS")
		# print("KEYPOINTS 1")
		# print(keypoints1)
		# print("\n")
		# print("KEYPOINTS 2")
		# print(keypoints2)
		
		if args.frame:
			frame_index = args.frame
		frame_image = cv2.imread(os.path.join(TEN_FPS_VIEWS_DIR, view, "images", 
								args.video, get_frame_image_filename(frame_index)))
		
		# print(os.path.join(TEN_FPS_VIEWS_DIR, view, "images", 
		# 						args.video, get_frame_image_filename(frame_index)))

		try:
			assignment, croppedA, croppedB = crop_and_assign(args.color, args.distance, keypoints1, keypoints2, 
													frame_image, histA, histB, actorA, actorB)
		except TypeError:
			continue

		keypointsA = [key for key, value in assignment.items() if value == 'A'][0]
		if keypointsA == 1:
			keypointsA = keypoints1
			keypointsB = keypoints2
		else:
			keypointsA = keypoints2
			keypointsB = keypoints1
		

		if args.frame:
			plt.imshow(croppedA)
			plt.title("A")
			print("A")
			print("X coords")
			print(min([ coord for coord in keypointsA[:,0] if coord != 0]), keypointsA[:,0].max())
			print("Y coords")
			print(min([ coord for coord in keypointsA[:,1] if coord != 0]), keypointsA[:,1].max())
			print("\n")
			plt.show()
			plt.title("B")
			plt.imshow(croppedB)
			print("B")
			print("X coords")
			print(min([ coord for coord in keypointsA[:,0] if coord != 0]), keypointsA[:,0].max())
			print("Y coords")
			print(min([ coord for coord in keypointsA[:,1] if coord != 0]), keypointsA[:,1].max())
			plt.show()

		write_cropped_images('tmp', frame_index, croppedA, croppedB)
		