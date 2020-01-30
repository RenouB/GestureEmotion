import sys
import os
import numpy
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from process_utils import filter_keypoints, get_crop_coordinates, get_frame_image_filename, write_cropped_images, \
convert_keypoints_to_array, add_keypoints_to_sequences, construct_reference_histograms, get_all_channels_hist, \
construct_reference_histograms

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

ACTOR_REFERENCE_IMAGES_DIR = constants["ACTOR_REFERENCE_IMAGES_DIR"]
PROJECT_DIR = constants["PROJECT_DIR"]


"""
visualize some image distance comparisons between color histograms for different images
"""

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-distance", default='cor', type=str)
	parser.add_argument("-only_hue", action="store_true", default=True)
	parser.add_argument("-color", default="hsv", type=str)
	parser.add_argument("-pair", default=0,type=int)
	parser.add_argument('-hist_diff', default=False)
	parser.add_argument("-num_bins", default=32, type=int)
	args = parser.parse_args()
	print(args)
	TEST_IMAGE_DIR = "test_images"
	REFERENCE_IMAGE_DIR = "../../annos_website/actor_ids"
	reference_hists, reference_indices = construct_reference_histograms(args.color, \
																args.only_hue, args.num_bins, args.hist_diff)
	
	pair_to_actor_ids = {
						 7: { 'A': '07', 'B': '08'},
						 9: { 'A': '09', 'B':'10' },
						 11:{ 'A': '11', 'B': '12' }
						 }

	actorA = pair_to_actor_ids[args.pair]['A']
	actorB = pair_to_actor_ids[args.pair]['B']

	actor_a_images = [image.path for image in os.scandir(TEST_IMAGE_DIR) 
						if actorA in image.path]

	actor_b_images = [image.path for image in os.scandir(TEST_IMAGE_DIR)
	 					if actorB in image.path]
	reference_a_hist = reference_hists[actorA]
	reference_b_hist = reference_hists[actorB]
	reference_a_image = cv2.imread('../../annos_website/actor_ids/'+actorA+'.png')
	reference_b_image = cv2.imread('../../annos_website/actor_ids/'+actorB+'.png')
	reference_a_image = cv2.cvtColor(reference_a_image, cv2.COLOR_BGR2RGB)
	reference_b_image = cv2.cvtColor(reference_b_image, cv2.COLOR_BGR2RGB)

	for images in [actor_a_images, actor_b_images]:
		print(images)

		fig = plt.figure()
		# each actor has two images. each image is comapred to two references.
		# results in 3x2 plot.
		# left side: test image
		# middle: actor a reference
		# right actor b reference
		
		# messy quick hack to get another sample image that isn't current one

		for i, image in enumerate(images):
			other_a = [im for im in actor_a_images if image != im][0]
			other_b = [im for im in actor_b_images if image != im][0]

			if args.color == 'hsv':
				sample_im, other_a_im, other_b_im = [cv2.imread(image) for image 
											in [image, other_a, other_b]]
				sample_im, other_a_im, other_b_im = [cv2.cvtColor(im, cv2.COLOR_BGR2HSV) for im 
													in [sample_im, other_a_im, other_b_im]]
				if args.only_hue == True:
					sample_hist, other_a_hist, other_b_hist = \
									[cv2.calcHist(im, [0], None, [min(255,args.num_bins)],
									[0,255]) for im in [sample_im, other_a_im, other_b_im]]
				else:
					sample_hist, other_a_hist, other_b_hist = \
							[get_all_channels_hist(im, min(args.num_bins, 255), 255)
							for im in [sample_im, other_a_im, other_b_im]]
				
				sample_im, other_a_im, other_b_im = [cv2.cvtColor(im, cv2.COLOR_HSV2RGB) for im
													in [sample_im, other_a_im, other_b_im]]
	
			elif args.color == 'rgb':
				sample_hist, other_a_hist, other_b_hist = \
								[get_all_channels_hist(im, args.num_bins, 256) 
								for im in [sample_im, other_a_im, other_b_im]]
			elif args.color == 'gray':					
				sample_hist, other_a_hist, other_b_hist = \
								[get_all_channels_hist(im, args.num_bins, 256) 
								for im in [sample_im, other_a_im, other_b_im]]


			if args.distance == 'cor':
				distance = cv2.HISTCMP_CORREL
				reverse = False
			elif args.distance == 'chi':
				distance = cv2.HISTCMP_CHISQR
				reverse = True
			elif args.distance == "intersect":
				distance = cv2.HISTCMP_INTERSECT
				reverse = False

			sample_hist, other_a_hist, other_b_hist = \
					[cv2.normalize(hist, hist).flatten() for hist in
					[sample_hist, other_a_hist, other_b_hist]]
			
			sim_refA = cv2.compareHist(sample_hist, reference_a_hist, distance)*100
			sim_refB = cv2.compareHist(sample_hist, reference_b_hist, distance)*100
			sim_otherA = cv2.compareHist(sample_hist, other_a_hist, distance)*100
			sim_otherB = cv2.compareHist(sample_hist, other_b_hist, distance)*100
			
			print("PLOTTING {}".format(image))
			# plot sample image
			ax = fig.add_subplot(2, 5, i*5+1)
			plt.imshow(sample_im)
			ax.set_title('sample {}'.format(i))
			plt.axis("off")

			# plot ref A
			ax = fig.add_subplot(2, 5, i*5+2)
			ax.set_title("ref A, {:.2f}".format(sim_refA))
			plt.imshow(reference_a_image)
			plt.axis("off")

			# plot ref B
			ax = fig.add_subplot(2, 5, i*5+3)
			ax.set_title("ref-B, {:.2f}".format(sim_refB))
			plt.imshow(reference_b_image)
			plt.axis("off")

			# plot other sample A
			ax = fig.add_subplot(2, 5, i*5+4)
			ax.set_title("samp-A, {:.2f}".format(sim_otherA))
			plt.imshow(other_a_im)
			plt.axis("off")
			
			# plot a sampleB
			ax = fig.add_subplot(2, 5, i*5+5)
			ax.set_title("samp-B, {:.2f}".format(sim_otherB))
			plt.imshow(other_b_im)
			plt.axis("off")

		fig.tight_layout()
		plt.show()