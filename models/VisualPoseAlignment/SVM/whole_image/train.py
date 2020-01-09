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
from definitions import constants, cnn_params
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from pretty_logging import PrettyLogger
from evaluation import get_scores

RANDOM_SEED = 200

HISTOGRAMS_DATA_DIR = constants["HISTOGRAMS_DATA_DIR"]


def get_scores(labels, predictions, detailed):
	micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(labels, predictions,
			 average='micro')
	macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(labels, predictions, 
			average='macro')
	prfs_per_class = precision_recall_fscore_support(labels, predictions, average=None)
	
	overall_acc = sum(labels == predictions) / len(predictions)

	scores = {'micro_p':micro_p, 'micro_r':micro_r, 'micro_f':micro_f,
			'macro_p': macro_p, 'macro_r': macro_r, 'macro_f':macro_f, 'exact_acc': overall_acc}
	
	if detailed:
		for i in np.unique(labels):
			i = int(i)
			scores[i - 1] = {}
			scores[i - 1]['p'] = prfs_per_class[0][i - 1]
			scores[i - 1]['r'] = prfs_per_class[1][i - 1]
			scores[i - 1]['f'] = prfs_per_class[2][i - 1]

	return scores

if __name__ == "__main__":
	
	parser = ArgumentParser()
	parser.add_argument('-color', default='hsv')
	parser.add_argument('-num_bins', default=32, type=int)
	parser.add_argument('-only_hue', action="store_true", default=False)

	parser.add_argument('-c', default=0.001, type=float)
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

	pair_scores = {}
	svms = {}
	pair_ids = list(train.keys())

	all_test_labels  = np.array([])
	all_test_predictions = np.array([])

	for pair_id in pair_ids:
		print("PROCESSING {}".format(pair_id))
		
		actorA = int(pair_id[:2])
		actorB = int(pair_id[2:])
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
		
		scores = get_scores(test_labels, predictions, detailed=True)
		test_labels[np.where(test_labels == 1)] = actorB
		test_labels[np.where(test_labels == 0)] = actorA
		predictions[np.where(predictions == 1)] = actorB
		predictions[np.where(predictions == 0)] = actorA
		all_test_labels = np.concatenate([all_test_labels, test_labels])
		all_test_predictions = np.concatenate([all_test_predictions, predictions])
		pair_scores[pair_id] = scores
		
		logger.update_scores(scores, pair_id, 'DEV')

	print(all_test_labels.shape)
	print(all_test_predictions.shape)
	"ARE THERE ANY FUCKINGS ONES?"
	print(sum(all_test_labels == 1))
	print(np.unique(all_test_labels))

	scores = get_scores(all_test_labels, all_test_predictions, detailed=True)
	pair_scores['all'] = scores
	logger.update_scores(scores, 'ALL', 'DEV')
	logger.close(0, 0)
	with open(os.path.join('./outputs/scores', basename+'scores.csv'), 'w+') as f:
		f.write(classification_report(all_test_labels, all_test_predictions))