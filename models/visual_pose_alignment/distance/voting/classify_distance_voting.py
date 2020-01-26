import os
import sys
import pickle
import numpy as np
import cv2
import time
from argparse import ArgumentParser

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)

from models.visual_pose_alignment.data.generate_histograms import convert_to_histogram
from models.evaluation import get_scores, update_scores_per_fold, \
average_scores_across_folds
import logging
from models.pretty_logging import PrettyLogger
import time

HISTOGRAMS_DATA_DIR = constants["HISTOGRAMS_DATA_DIR"]
MPIIEMO_ANNOS_WEBSITE = constants["MPIIEMO_ANNOS_WEBSITE"]


''' 
This script will assigns poses to actors by 
comparing the color histogram of a cropped body image
to  reference color histograms of the actors. Datapoints 
are cut into sub-images and classified using majority vote.

Many comments from this script apply to other training scripts in visual_pose_alignment
'''

if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument('-color', default='hsv', help="color channels to use")
	parser.add_argument('-num_bins', default=32, type=int, 
						help="histogram binning strategy")
	parser.add_argument('-only_hue', action="store_true", default=True, 
						help="if using hsv color scheme, can use only hiue")
	parser.add_argument('-distance', default = "cor", 
						help="distance metric for calculating similarity")
	args = parser.parse_args()
	basename = '-'.join([args.color, 'only_hue', str(args.only_hue),
					 str(args.num_bins)])+'-'
	
	starttime = time.strftime('%H%M-%b-%d-%Y')
	logs_dir = './outputs/logs'
	# initialize logger
	logger = PrettyLogger(args, logs_dir, basename, starttime)

	# assign distance measure for similarity calculation
	if args.distance == 'cor':
		distance = cv2.HISTCMP_CORREL
		reverse = False
	elif args.distance == 'chi':
		distance = cv2.HISTCMP_CHISQR
		reverse = True
	elif args.distance == "intersect":
		distance = cv2.HISTCMP_INTERSECT
		reverse = False
	
	# load test data
	with open(os.path.join(HISTOGRAMS_DATA_DIR, basename+'test.pkl'), 'rb') as f:
		test = pickle.load(f)

	
	# get list of unique actor pair ids
	pair_ids = list(test.keys())
	# create reference histograms using images of each actor
	reference_histograms = {}
	for reference_image in os.scandir(os.path.join(MPIIEMO_ANNOS_WEBSITE, 'actor_ids')):
		actor_id = reference_image.name[:-4]
		im = cv2.imread(reference_image.path)
		reference_histograms[actor_id] = convert_to_histogram(im, args.num_bins,
											args.color, args.only_hue)
	
	# initialize variable to keep track of scores for each actor pair
	# k is not a fold. simply counts unique actor pairs.
	# all train scores are dummies - we won't need to train.
	scores_per_fold = {'train':{},'dev':{}}
	k = -1
	for pair_id in pair_ids:
		k += 1

		scores_per_fold['train'][k] = {'macro':[[0,0,0]], 0:[[0,0,0]], 
				1:[[0,0,0]], 'loss':[[0]], 'att_weights':[[0,0,0]], 'acc':[[0]]}
		scores_per_fold['dev'][k] = {'macro':[], 0:[], 1:[], 
				'loss': [], 'att_weights':[], 'acc':[]}

		print("PROCESSING {}".format(pair_id))
		
		# get actorA, actorB ids from pair id
		actorA = pair_id[:2]
		actorB = pair_id[2:]
		# test X: num datapoints * num histogram bins * 9 image subregions
		test_X = test[pair_id]['hists'].squeeze(3)
		test_labels = test[pair_id]['labels'][:,0]
		
		predictions = []
		for i, row in enumerate(test_X):
			# preds will house sub-region predictions, which are subsequently used for
			# image prediction
			preds = []
			# reshape to get each sub-region histogram along rows
			test_x = row.reshape(9,-1)
			for sub_region_hist in test_x:
				# compute similarities between sub_region, reference images
				simA = cv2.compareHist(sub_region_hist, reference_histograms[actorA], distance)
				simB = cv2.compareHist(sub_region_hist, reference_histograms[actorB], distance)
				
				# for chi, smaller similarity measure is better
				if reverse:
					if simA < simB:
						preds.append(0)
					else:
						preds.append(1)
				
				else:
					if simA > simB:
						preds.append(0)
					else:
						preds.append(1)
			preds = np.array(preds)

			# classify by majority vote in preds
			num_ones = sum(preds == 1)
			num_zeroes = sum(preds == 0)

			if num_ones > num_zeroes:
				predictions.append(1)
			elif num_ones == num_zeroes:
				predictions.append(np.random.choice([0,1],1).item())
			else:
				predictions.append(0)

		predictions = np.array(predictions)
		scores = get_scores(test_labels, predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'dev',
							0, [0,0,0], len(test_labels), k)
	
		logger.update_scores(scores, pair_id, 'DEV')
		
		
	# scoring
	av_scores = average_scores_across_folds(scores_per_fold)
	scores = {'average': av_scores, 'all':scores_per_fold}
	best_epoch = np.amax(av_scores['dev'][1][:,2]).astype(np.int)
	macro_scores = av_scores['dev']['macro'][best_epoch]

	with open(os.path.join('outputs', 'scores', basename+'.pkl'), 'wb') as f:
		pickle.dump(scores, f)

	with open(os.path.join('outputs', 'scores', basename+'.csv'), 'a+') as f:
		f.write("BEST EPOCH: {} \n".format(best_epoch))
		f.write("{:>8} {:>8} {:>8} {:>8} {:>8}\n".format("class", "p", "r", "f", "acc"))
		f.write("{:8} {:8.4f} {:8.4f} {:8.4f} {:8.4f} \n".format("macro",
				macro_scores[0], macro_scores[1], macro_scores[2], av_scores['dev']['acc'][best_epoch][0]))
	
	logger.close(0, 0)