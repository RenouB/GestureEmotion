import sys
import os
import numpy
import json
import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from argparse import ArgumentParser


PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from models.visual_pose_alignment.data.generate_histograms import convert_to_histogram
from definitions import constants
from MPIIEmo.prepare_data.visual_keypoint_alignment.process_utils import filter_keypoints, get_crop_coordinates, get_body_image_filename, \
convert_keypoints_to_array, crop, add_keypoints_to_sequences, normalize_keypoints, \
interpolate_keypoints_all_frames, interpolate_missing_coordinates

MANUALLY_SELECTED_IMAGES_DIR = constants["MANUALLY_SELECTED_IMAGES_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]

def get_average_distance(h, last_hists, hist, distance):
	total_distance = 0
	for previous_hist in last_hists[min(0, len(last_hists) - 10):]:
		total_distance += cv2.compareHist(previous_hist, hist, distance)
	return total_distance

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-color', default='hsv', help="hsv, rgb or grab")
	parser.add_argument('-only_hue', action="store_true", default=False, 
						help="use only hue from hsv")
	parser.add_argument('-num_bins', default=32, type=int,
						help="num bins for histograms")
	parser.add_argument('-history', default=10)
	parser.add_argument('-distance', default='cor')
	args = parser.parse_args()
	train = {}
	test = {}
	all_together = {'views':[], 'actor_pairs':[], 'labels':[], 'predictions':[], 'keypoints' :[]}
	for split, folder in [(train, 'train'), (test, 'test')]:
		for view in os.scandir(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder)):
			for video in os.scandir(view.path):
				last_actorA_hists = []
				last_actorB_hists = []

				video_name = video.name
				actor_pair = video_name.split('_')[2][1:]+video.name.split('_')[3][1:]
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
					
	
					actorA_images = sorted(os.listdir(os.path.join(video.path, 'A')))
					actorB_images = sorted(os.listdir(os.path.join(video.path, 'B')))
		
					path1 = get_body_image_filename(frame_index, 1)
					path2 = get_body_image_filename(frame_index, 2)
					# print(path1, path2, '\n',actorA_images, '\n',actorB_images)
					im1 = None
					if path1 in actorA_images:
						im1 = cv2.imread(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder, view.name,
										 video.name, 'A', path1))		
						
						all_together['labels'].append(0)

					elif path1 in actorB_images:
						if path1 in actorA_images:
							im1 = cv2.imread(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder, view.name,
											 video.name, 'B', path1))		
							all_together['labels'].append(1)
					if im1 is None:
						all_together['labels'].append('none')
					
					im2 = None
					if path2 in actorA_images:
						im2 = cv2.imread(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder, view.name,
										 video.name, 'A', path2))		
						all_together['labels'].append(0)
					elif path2 in actorB_images:
						if path2 in actorA_images:
							im2 = cv2.imread(os.path.join(MANUALLY_SELECTED_IMAGES_DIR, folder, view.name,
											 video.name, 'B', path2))		
							all_together['labels'].append(1)
					if im2 is None:
						all_together['labels'].append('none')

					if im1 is not None:
						hist1 = convert_to_histogram(im1, args.num_bins, 
									args.color, args.only_hue)
					if im2 is not None:
						hist2 = convert_to_histogram(im2, args.num_bins,
									args.color, args.only_hue)


					# here rule based stuff to get things started					
					if type(keypoints1) == str and keypoints1 == 'tooclose':						
						one_pred = 'none'
						two_pred = 'none'

					elif len(last_actorA_hists) == 0 and len(last_actorB_hists) == 0:
						if type(keypoints2) == str:
							one_pred = 1
							last_actorB_hists.append(hist1)
							two_pred = 'none'

						elif type(keypoints1) == np.ndarray and type(keypoints2 == np.ndarray):
							if view.name == 'view2':
							# if its beginning of video and alreayd have two keypoints
							# rightmost is B
								one_max_x = keypoints1[0,:].max()
								two_max_x = keypoints2[0,:].max()
								if one_max_x > two_max_x:
									one_pred = 1
									two_pred = 0
									last_actorA_hists.append(hist2)
									last_actorB_hists.append(hist1)
								elif two_max_x > one_max_x:
									two_pred = 1
									one_pred = 0
									last_actorA_hists.append(hist1)
									last_actorB_hists.append(hist2)
							elif view.name == 'view6':
								# in view 6, leftmost person at beginning is B
								one_min_x = keypoints1[0,:].min()
								two_min_x = keypoints2[0,:].min()
								if one_min_x < two_min_x:
									one_pred = 1
									two_pred = 0
									last_actorA_hists.append(hist2)
									last_actorB_hists.append(hist1)
								elif two_min_x < one_min_x:
									one_pred = 0
									two_pred = 1
									last_actorA_hists.append(hist1)
									last_actorB_hists.append(hist2)
					
					elif len(last_actorA_hists) == 0 and keypoints2 == 'none':
						one_pred = 1
						two_pred = 0
						last_actorB_hists.append(hist1)
						if args.distance == 'cor':
							distance = cv2.HISTCMP_CORREL
							reverse = False
						elif args.distance == 'chi':
							distance = cv2.HISTCMP_CHISQR
							reverse = True
						elif args.distance == "intersect":
							distance = cv2.HISTCMP_INTERSECT
							reverse = False
					# all begining conditions should have been taken care of.
					else:
						if im1 is not None:
							sim1_A = get_average_distance(args.history, last_actorA_hists, 
											hist1, distance)
							sim1_B = get_average_distance(args.history, last_actorB_hists, 
											hist1, distance)
							if reverse:
								sim1_A = sim1_A*-1
								sim1_B = sim1_B*-1
							
							if sim1_A > sim1_B:
								one_pred = 0
								last_actorA_hists.append(hist1)
								if im2 is not None:
									two_pred = 1
									last_actorB_hists.append(hist2)
								else:
									two_pred = 'none'

						elif im2 is not None:
							sim2_A = get_average_distance(args.history, last_actorA_hists, 
											hist2, distance)
							sim2_B = get_average_distance(args.history, last_actorB_hists, 
											hist2, distance)
							if reverse:
								sim2_A = sim2_A*-1
								sim2_B = sim2_B*-1

							if sim2_A > sim2_B:
								two_pred = 0
								last_actorA_hists.append(hist2)
								if im1 is not None:
									one_pred = 1
									last_actorB_hists.append(hists)
								else:
									one_pred ='none'
					# now I need to start looking

					print(frame_index, video.name)
					print(one_pred, two_pred)
					all_together['predictions'].append(one_pred)
					all_together['predictions'].append(two_pred)
					all_together['views'].append(view.name)
					all_together['views'].append(view.name)
					all_together['actor_pairs'].append(actor_pair)
					all_together['actor_pairs'].append(actor_pair)


indices_to_keep = [i for i in range(len(all_together['labels'])) if type(all_together['labels'][i]) != str
					and type(all_together['predictions'][i]) != str]
print(all_together['labels'][300:400])
print(all_together['predictions'][300:400])
labels = np.array([all_together['labels'][i] for i in indices_to_keep])
predictions = np.array([all_together['predictions'][i] for i in indices_to_keep])
actor_pairs = [all_together['actor_pairs'][i] for i in indices_to_keep]
print(labels.shape, predictions.shape)
print(sum(labels == predictions))
