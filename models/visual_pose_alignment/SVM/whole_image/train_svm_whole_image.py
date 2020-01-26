import os
import sys
import pickle
import numpy as np
import cv2
import time
from argparse import ArgumentParser
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support, classification_report
from sklearn.svm import LinearSVC


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

RANDOM_SEED = 200

"""
For each actor pair, a unique SVM is trained to assign poses to actors
by taking color histograms of cropped body images as input.
"""

if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument('-color', default='hsv', 
						help="color channels to use")
	parser.add_argument('-num_bins', default=32, type=int, 
						help="histogram binning strategy")
	parser.add_argument('-only_hue', action="store_true", default=True, 
						help="if using hsv color scheme, can use only hiue")
	parser.add_argument('-C', default=0.001, type=float)
	args = parser.parse_args()
	
	basename = '-'.join([args.color, 'only_hue', str(args.only_hue),
					 str(args.num_bins)])+'-'
	
	starttime = time.strftime('%H%M-%b-%d-%Y')
	logs_dir = './outputs/logs'
	logger = PrettyLogger(args, logs_dir, basename, starttime)


	with open(os.path.join(HISTOGRAMS_DATA_DIR, basename+'train.pkl'), 'rb') as f:
		train = pickle.load(f)
	with open(os.path.join(HISTOGRAMS_DATA_DIR, basename+'test.pkl'), 'rb') as f:
		test = pickle.load(f)

	pair_ids = list(train.keys())

	scores_per_fold = {'train':{},'dev':{}}
	k = -1
	for pair_id in pair_ids:
		k += 1
		scores_per_fold['train'][k] = {'macro':[[0,0,0]], 0:[[0,0,0]], 1:[[0,0,0]], 'loss':[[0]], 'att_weights':[[0,0,0]], 'acc':[[0]]}
		scores_per_fold['dev'][k] = {'macro':[], 0:[], 1:[], 'loss': [], 'att_weights':[], 'acc':[]}

		print("PROCESSING {}".format(pair_id))
		
		clf = LinearSVC(random_state = RANDOM_SEED, tol=0.00001, C=args.c)
		
		X = train[pair_id]['hists'][:,0,:,:].squeeze(2)
		y = train[pair_id]['labels'][:,0]
		print("shape X: {} shape y: {}".format(X.shape, y.shape))
		print("FITTING")
		
		clf.fit(X, y)
		
		test_X = test[pair_id]['hists'][:,0,:,:].squeeze(2)
		test_labels = test[pair_id]['labels'][:,0]
		print("PREDICTING")
		print("shape X: {} shape labels: {}".format(test_X.shape, test_labels.shape))
		predictions = clf.predict(test_X)
		
		predictions = np.array(predictions)
		scores = get_scores(test_labels, predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'dev',
							0, [0,0,0], len(test_labels), k)
		logger.update_scores(scores, pair_id, 'DEV')
		
	
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