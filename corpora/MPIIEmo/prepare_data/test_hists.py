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

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

ACTOR_REFERENCE_IMAGES_DIR = constants["ACTOR_REFERENCE_IMAGES_DIR"]


def get_all_channels_hist(image, max_range):
	hist0 = cv2.calcHist(image, [0], None, [max_range], [0,max_range])
	hist1 = cv2.calcHist(image, [1], None, [max_range], [0,max_range])
	hist2 = cv2.calcHist(image, [2], None, [max_range], [0,max_range])

	return np.concatenate([hist0, hist1, hist2], axis=0)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-dark", type=str)
	parser.add_argument("-light", type=str)
	parser.add_argument("-distance", default='cor', type=str)
	parser.add_argument("-only_hue", action="store_true", default=False)
	parser.add_argument("-color", default="hsv", type=str)
	parser.add_argument("-ref1", default='09', type=str)
	parser.add_argument("-ref2", default='10', type=str)
	args = parser.parse_args()
	print(args)
	dark_path = os.path.join('tmp', 'dark', args.dark+'.png')
	light_path = os.path.join('tmp', 'light', args.light+'.png')
	
	dark = cv2.imread(dark_path)
	light = cv2.imread(light_path)
	
	ref1 = cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, args.ref1+'.png'))
	ref2 = cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, args.ref2+'.png'))

	if args.color == 'gray':
		dark = cv2.cvtColor(dark, cv2.COLOR_BGR2GRAY)
		light = cv2.cvtColor(light, cv2.COLOR_BGR2GRAY)
		ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2GRAY)
		ref2 = cv2.cvtColor(ref2, cv2.COLOR_BGR2GRAY)		

		dark_hist = cv2.calcHist(dark, [0], None, [32], [0,256])
		print(dark_hist)
		light_hist = cv2.calcHist(light, [0], None, [32], [0,256])
		ref1_hist = cv2.calcHist(ref1, [0], None, [32], [0,256])
		ref2_hist = cv2.calcHist(ref2, [0], None, [32], [0,256])

	if args.color == 'rgb':
		dark_hist = get_all_channels_hist(dark, 256)
		light_hist = get_all_channels_hist(light, 256) 
		ref1_hist = get_all_channels_hist(ref1, 256) 
		ref2_hist = get_all_channels_hist(ref2, 256) 


	if args.color == 'hsv':
		dark = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)
		light = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
		ref1 = cv2.cvtColor(ref1, cv2.COLOR_BGR2HSV)
		ref2 = cv2.cvtColor(ref2, cv2.COLOR_BGR2HSV)

		if args.only_hue:

			dark_hist = cv2.calcHist(dark, [0], None, [255], [0,255])
			light_hist = cv2.calcHist(light, [0], None, [255], [0,255])
			ref1_hist = cv2.calcHist(ref1, [0], None, [255], [0,255])
			ref2_hist = cv2.calcHist(ref2, [0], None, [255], [0,255])

		else:
			dark_hist = get_all_channels_hist(dark, 255)
			light_hist = get_all_channels_hist(light, 255) 
			ref1_hist = get_all_channels_hist(ref1, 255) 
			ref2_hist = get_all_channels_hist(ref2, 255) 

	print(dark_hist)
	dark_hist = cv2.normalize(dark_hist, dark_hist).flatten()
	print(dark_hist)
	light_hist = cv2.normalize(light_hist, light_hist).flatten()
	ref1_hist = cv2.normalize(ref1_hist, ref1_hist).flatten()
	ref2_hist = cv2.normalize(ref2_hist, ref2_hist).flatten()

	if args.distance == 'cor':
		distance = cv2.HISTCMP_CORREL
		reverse = False
	elif args.distance == 'chi':
		distance = cv2.HISTCMP_CHISQR
		reverse = True
	elif args.distance == "intersect":
		distance = cv2.HISTCMP_INTERSECT

	sim_dark_ref1 = cv2.compareHist(dark_hist, ref1_hist, distance)
	sim_dark_ref2 = cv2.compareHist(dark_hist, ref2_hist, distance)

	sim_light_ref1 = cv2.compareHist(light_hist, ref1_hist, distance)
	sim_light_ref2 = cv2.compareHist(light_hist, ref2_hist, distance)

	fig = plt.figure()

	ax = fig.add_subplot(2, 3, 1)
	ax.set_title('comparison im')
	plt.imshow(dark)
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 2)
	ax.set_title("{:.2f} {}".format(sim_dark_ref1, str(distance)))
	plt.imshow(ref1)
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 3)
	ax.set_title("{:.2f} {}".format(sim_dark_ref2, str(distance)))
	plt.imshow(ref2)
	plt.axis("off")


	ax = fig.add_subplot(2, 3, 4)
	ax.set_title('comparison im')
	plt.imshow(light)
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 5)
	ax.set_title("{:.2f} {}".format(sim_light_ref1, distance))
	plt.imshow(ref1)
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 6)
	ax.set_title("{:.2f} {}".format(sim_light_ref2, distance))
	plt.imshow(ref2)
	plt.axis("off")

	plt.show()