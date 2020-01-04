import sys
import os
import numpy
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from process_utils2 import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images, \
convert_keypoints_to_array, add_keypoints_to_sequences, construct_reference_histograms, get_all_channels_hist, \
construct_reference_histograms

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

ACTOR_REFERENCE_IMAGES_DIR = constants["ACTOR_REFERENCE_IMAGES_DIR"]
PROJECT_DIR = constants["PROJECT_DIR"]

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-distance", default='cor', type=str)
	parser.add_argument("-only_hue", action="store_true", default=False)
	parser.add_argument("-color", default="hsv", type=str)
	parser.add_argument("-plot", action="store_true", default=False)
	parser.add_argument("-print_all", action="store_true", default=False)
	parser.add_argument("-hist_diff", action="store_true", default=False)
	parser.add_argument("-num_bins", default=256, type=int)
	args = parser.parse_args()
	print(args)
	TEST_DIR = os.path.join(PROJECT_DIR, 'corpora/MPIIEmo/prepare_data/visual-keypoint-alignment/tmp')
	
	reference_hists, reference_indices = construct_reference_histograms(args.color, \
																args.only_hue, args.num_bins, args.hist_diff)
	# loop over each folder, get list of images in each actor folder
	# load them, get histogram, and compare them to reference
	# print metrics
	# count if it was right or wrong

	num_comparisons = 0
	num_correct = 0

	for video in os.listdir(TEST_DIR):
		video_comparisons = 0
		video_correct = 0
		video_folder = os.path.join(TEST_DIR, video)
		actor_images = {}



		
		for actor in [item for item in os.listdir(video_folder) if item.isnumeric()]:
			actor_folder = os.path.join(video_folder, actor)
			actor_images[actor] = os.listdir(os.path.join(video_folder, actor))

		max_length = min([len(images_list) for images_list in actor_images.values()])
		
		actor_ids = []
		actor_imgs = []
		for key, value in actor_images.items():
			actor_ids.append(key)
			actor_imgs.append(value[:max_length])


		first_id = actor_ids[0]
		second_id = actor_ids[1]
		ref1 = reference_hists[first_id]
		ref2 = reference_hists[second_id]
		indices_to_compare = reference_indices[first_id]

		first_write_dir = os.path.join(video_folder, first_id+'classified')
		second_write_dir = os.path.join(video_folder, second_id+'classified')

		os.system("rm -rf {}/*".format(first_write_dir))
		os.system("rm -rf {}/*".format(second_write_dir))
		os.system("mkdir {}".format(first_write_dir))
		os.system("mkdir {}".format(second_write_dir))

		print("VIDEO {}".format(video))
		print("Len ref indices:", len(indices_to_compare))
		for i in range(len(actor_imgs[0])):
			first_image = actor_imgs[0][i]
			second_image = actor_imgs[1][i]
			first_image = cv2.imread(os.path.join(video_folder, first_id, first_image))
			second_image = cv2.imread(os.path.join(video_folder, second_id, second_image))

			if args.color == 'hsv':
				first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2HSV)
				second_image = cv2.cvtColor(second_image, cv2.COLOR_BGR2HSV)
				
				if args.only_hue == True:
					first_hist = cv2.calcHist(first_image, [0], None, [min(255,args.num_bins)], [0,255])
					second_hist = cv2.calcHist(second_image, [0], None, [min(255,args.num_bins)], [0,255])
				else:
					first_hist = get_all_channels_hist(first_image, min(args.num_bins, 255), 255)
					second_hist = get_all_channels_hist(second_image, min(args.num_bins, 255), 255)

			elif args.color == 'rgb':
				first_hist = get_all_channels_hist(first_image, args.num_bins, 256)
				second_hist = get_all_channels_hist(second_image, args.num_bins, 256)

			elif args.color == 'gray':					
				first_hist = get_all_channels_hist(first_image, args.num_bins, 256)
				second_hist = get_all_channels_hist(second_image, args.num_bins, 256)

			if args.distance == 'cor':
				distance = cv2.HISTCMP_CORREL
				reverse = False
			elif args.distance == 'chi':
				distance = cv2.HISTCMP_CHISQR
				reverse = True
			elif args.distance == "intersect":
				distance = cv2.HISTCMP_INTERSECT
				reverse = False

			first_hist = cv2.normalize(first_hist, first_hist).flatten()
			second_hist = cv2.normalize(second_hist, second_hist).flatten()

			first_hist = np.array([first_hist[i] for i in indices_to_compare])
			second_hist = np.array([second_hist[i] for i in indices_to_compare])

			first_sim_first = cv2.compareHist(first_hist, ref1, distance)
			first_sim_second = cv2.compareHist(first_hist, ref2, distance)
			second_sim_first = cv2.compareHist(second_hist, ref1, distance)
			second_sim_second = cv2.compareHist(second_hist, ref2, distance)

			if args.print_all:
				print("FIRST {}".format(first_id))
				print("SIM FIRST: {:.2f} SIM SECOND: {:.2f}".format(first_sim_first, first_sim_second))

				print("SECOND {}".format(second_id))
				print("SIM FIRST: {:.2f} SIM SECOND: {:.2f}".format(second_sim_first, second_sim_second))
				print("\n")


			print("############")
			print("skeleton one")
			print(first_hist)
			print("skeleton two")
			print(second_hist)
			print("ref1")
			print(ref1)
			print("ref2")
			print(ref2)

			if args.plot:
				fig = plt.figure()

				ax = fig.add_subplot(2, 3, 1)
				ax.set_title(first_id)
				plt.imshow(first_image)
				plt.axis("off")

				ax = fig.add_subplot(2, 3, 2)
				ax.set_title("{:.8f}".format(first_sim_first))
				plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, first_id+'.png')))
				plt.axis("off")

				ax = fig.add_subplot(2, 3, 3)
				ax.set_title("{:.8f}".format(first_sim_second))
				plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, second_id+'.png')))
				plt.axis("off")


				ax = fig.add_subplot(2, 3, 4)
				ax.set_title(second_id)
				plt.imshow(second_image)
				plt.axis("off")

				ax = fig.add_subplot(2, 3, 5)
				ax.set_title("{:.8f}".format(second_sim_first))
				plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, first_id+'.png')))
				plt.axis("off")

				ax = fig.add_subplot(2, 3, 6)
				ax.set_title("{:.8f}".format(second_sim_second))
				plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, second_id+'.png')))
				plt.axis("off")

				plt.show()

			if not reverse:
				best = max([first_sim_first, first_sim_second, second_sim_first, second_sim_second])
			else:
				best = min([first_sim_first, first_sim_second, second_sim_first, second_sim_second])

			if best == first_sim_first or best == first_sim_second and \
				first_sim_first != second_sim_first:
				num_correct += 1
				video_correct += 1			
			num_comparisons += 1
			video_comparisons += 1

			first_image = cv2.cvtColor(first_image, cv2.COLOR_HSV2BGR)
			second_image = cv2.cvtColor(second_image, cv2.COLOR_HSV2BGR)
			if best == first_sim_first or best == second_sim_second:
				cv2.imwrite(os.path.join(first_write_dir,str(i)+'.png'), first_image)
				cv2.imwrite(os.path.join(second_write_dir,str(i)+'.png'), second_image)
			elif best == first_sim_second or best == second_sim_first:
				cv2.imwrite(os.path.join(first_write_dir,str(i)+'.png'), second_image)
				cv2.imwrite(os.path.join(second_write_dir,str(i)+'.png'), first_image)
		print(print("percentage correct: {:.2f}".format(video_correct/video_comparisons)))
	print("percentage correct: {:.2f}".format(num_correct/num_comparisons))	