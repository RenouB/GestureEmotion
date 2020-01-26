import os
import sys
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from sklearn.metrics import multilabel_confusion_matrix

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

MANUALLY_SELECTED_IMAGES_DIR = constants["MANUALLY_SELECTED_IMAGES_DIR"]

TRAIN_DIR = os.path.join(MANUALLY_SELECTED_IMAGES_DIR, 'train')
TEST_DIR = os.path.join(MANUALLY_SELECTED_IMAGES_DIR, 'test')
WRITE_DIR = constants["HISTOGRAMS_DATA_DIR"]

"""
From manually sorted cropped body poses, construct a datasets of
color histograms using different parametres.


"""
train = {}
test = {}


def get_all_channels_hist(image, num_bins, max_range):
	"""
	image: cv2 image
	num_bins: binning strategy for histograms
	max_range: max value for color scheme

	return: concatenation of histograms from each of three color channels
	"""
	hist0 = cv2.calcHist(image, [0], None, [min(max_range, num_bins)], [0,max_range])
	hist1 = cv2.calcHist(image, [1], None, [min(max_range, num_bins)], [0,max_range])
	hist2 = cv2.calcHist(image, [2], None, [min(max_range, num_bins)], [0,max_range])

	return np.concatenate([hist0, hist1, hist2], axis=0)

def convert_to_histogram(im, num_bins, color, only_hue):
	"""
	im: cv2 image
	num_bins: binning strategy for histograms
	color: rgb, hsv or gray
	only_hue: if hsv, use only hue channel

	return: normalized histogram
	"""
	if color == 'hsv':
		color = cv2.COLOR_BGR2HSV
		im = cv2.cvtColor(im, color)
		if only_hue:
			hist = cv2.calcHist(im, [0], None, [num_bins], [0,256])
		else:
			hist = get_all_channels_hist(im, num_bins, 256)
	elif color == 'rgb':
		hist = get_all_channels_hist(im, num_bins, 256)
	elif color == 'gray':
		color = cv2.COLOR_BRG2GRAY
		im = cv2.cvtColor(im, color)
		hist = cv2.calcHist(im, [0], None, [num_bins], [0,256])
	
	return cv2.normalize(hist, hist)

def split_image(image, num_bins, color, only_hue):
	"""
	cut image into different sub regions and get corresponding
	color histograms

	image: cv2 image
	num_bins: binning strategy for histograms
	color: hsv, rgb, gray
	only_hue: if hsv, use only hue

	return: list of color histograms for each sub region
	"""

	all_hists = []
	
	max_y, max_x = image.shape[:2]
	sub_images = [image]
	# top half
	sub_images.append(image[0:int(max_y / 2),:])
	# bottom half
	sub_images.append(image[int(max_y / 2): max_y,:])
	# left half
	sub_images.append(image[:,0:int(max_x / 2)])
	# right half
	sub_images.append(image[:,int(max_x / 2):max_x])
	# top left
	sub_images.append(image[0:int(max_y / 2), 0:int(max_x / 2)])
	# top right
	sub_images.append(image[0:int(max_y / 2), int(max_x / 2):max_x])
	# bottom left
	sub_images.append(image[int(max_y / 2):max_y, 0:int(max_x / 2)])
	# bottom right
	sub_images.append(image[int(max_y / 2):max_y, int(max_x / 2):max_x])

	return [convert_to_histogram(sub_image, num_bins, color, only_hue) \
			for sub_image in sub_images]

if __name__ == "__main__":
	"""
	construct histogram dataset according to different params, save to disk
	"""
	
	parser = ArgumentParser()
	parser.add_argument('-color', default='hsv', help="hsv, rgb or grab")
	parser.add_argument('-only_hue', action="store_true", default=False, 
						help="use only hue from hsv")
	parser.add_argument('-num_bins', default=32, type=int,
						help="num bins for histograms")
	args = parser.parse_args()

	# iterate over train and test data
	for split, folder in[(train, TRAIN_DIR), (test, TEST_DIR)]:
		for video_folder in os.scandir(folder):
			# iterate over video folders in train/test
			print(video_folder.path)
			# get actor IDs
			actorA = video_folder.name.split('_')[2][1:]
			actorB = video_folder.name.split('_')[3][1:-4]
			# construct actor pair ID
			pair_id = ''.join(sorted([actorA, actorB], key= lambda e: int(e)))


			if pair_id not in split:
				split[pair_id] = {'hists':[], 'labels':[], 'actor_ids':[]}

			# iterate over image folders for each actor
			for actor_folder in os.scandir(os.path.join(video_folder.path)):
				# iterate over each image in that folder
				for image in os.scandir(actor_folder.path):
					im = cv2.imread(image.path)
					# get histograms for whole image and image sub regions
					hists = split_image(im, args.num_bins, args.color, args.only_hue)
					# append actor IDs for each image sub region
					# map actor IDs to labels
					if actor_folder.name == 'A':
						split[pair_id]['actor_ids'].append([actorA]*len(hists))
						label = 0
					elif actor_folder.name == 'B':
						split[pair_id]['actor_ids'].append([actorB]*len(hists))
						label = 1

					# append color histograms, labels to corresponding
					# lists for each actor pair
					split[pair_id]['hists'].append(hists)
					split[pair_id]['labels'].append([label]*len(hists))

	# iterate over data again and convert everything to arrays
	for split in [train,test]:
		for pair_id, data in split.items():
			for key in data:
				split[pair_id][key] = np.array(split[pair_id][key])

	basename = '-'.join([args.color, 'only_hue', str(args.only_hue), str(args.num_bins)])+'-'
	
	with open(os.path.join(WRITE_DIR, basename+'train.pkl'), 'wb') as f:
		pickle.dump(train, f)

	with open(os.path.join(WRITE_DIR, basename+'test.pkl'), 'wb') as f:
		pickle.dump(test, f)

