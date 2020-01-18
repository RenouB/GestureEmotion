import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn_crfsuite import CRF
np.random.seed(200)
from sklearn.metrics import multilabel_confusion_matrix, classification_report

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants, cnn_params
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.data.torch_datasets import PoseDataset
from models.evaluation import get_scores
import argparse
import logging
from models.pretty_logging import PrettyLogger, construct_crf_basename, get_write_dir
import time

def get_input_dim(keypoints):
	if keypoints == 'full':
		return len(constants["WAIST_UP_BODY_PART_INDICES"]) * 2
	if keypoints == 'full-hh':
		return len(constants["FULL-HH"]) * 2
	if keypoints == 'full-head':
		return len(constants["FULL-HEAD"]) * 2
	if keypoints == 'head':
		return len(constants["HEAD"]) * 2
	if keypoints == 'hands':
		return len(constants["HANDS"]) * 2

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-joint', action="store_true", default=False)
	parser.add_argument('-modalities', default=0, type=int)
	parser.add_argument('-attention', default=False)
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_length', default=5, type=int)
	parser.add_argument("-emotion", default=0, type=int)
	parser.add_argument("-keypoints", default='full')

	parser.add_argument('-c1', type=float, default=0.01)
	parser.add_argument('-c2', type=float, default=0.01)
	parser.add_argument('-max_iterations', type=int, default=100)
	parser.add_argument('-algo', default='lbfgs')
	parser.add_argument('-num_folds', default=1, type=int)
	parser.add_argument('-test', action='store_true', default=False)
	parser.add_argument('-debug', action='store_true', default=False)
	parser.add_argument('-comment', default='')
	args = parser.parse_args()
	
	if args.debug:
		print_denominator = 5
	else:
		print_denominator = 100
	
	print("################################################")
	print("                  STARTING")
	
	# basename for logs, weights
	starttime = time.strftime('%H%M-%b-%d-%Y')
	basename = construct_crf_basename(args)+'-'+starttime
	write_dir = get_write_dir('CRF', attention=False, joint=args.joint, modalities=args.modalities, 
		emotion= args.emotion)
	print(write_dir)
	logger = PrettyLogger(args, os.path.join(write_dir, 'logs'), basename, starttime)

	# TODO: Add different modalities
	data = PoseDataset(args.interval, args.seq_length, args.keypoints, args.joint, args.debug, args.emotion)
	best_f1 = -1
	best_k = -1
	final_scores_per_fold = {}
	losses = {}
	f1s = {}

	
	for k in range(args.num_folds):
		f1s[k] = {'train:':[], 'dev':[]}
		logger.new_fold(k)
		train_indices, dev_indices = data.split_data(k, args.emotion)
		train_data = Subset(data, train_indices)
		train_loader =DataLoader(train_data, batch_size=len(train_indices), shuffle=True)
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=len(dev_indices))
		train_X = np.array([data[i]['pose'] for i in train_indices])
		train_y = np.array([data[i]['labels'] for i in train_indices])

		dev_X = np.array([data[i]['pose'] for i in dev_indices])
		dev_labels = np.array([data[i]['labels'] for i in dev_indices])

		print("################################################")
		print("                Beginning fold {}".format(k))
		print("            Length train data: {}".format(len(train_data)))
		print("              Length dev data: {}".format(len(dev_data)))
		print("################################################")
		print("                    TRAIN")
		print("################################################")
		
		
		crf = CRF(algorithm=args.algo, c1=args.c1, c2=args.c2,
	    max_iterations=args.max_iterations, all_possible_transitions=True )
		crf.fit(train_X, train_y)

		print("################################################")
		print("                     EVAL")
		print("################################################")
		
		dev_predictions = crf.predict(dev_X)
		scores = get_scores(dev_labels, dev_predictions, detailed=False)
		f1s[k]['dev'].append(scores['micro_f'])
		logger.update_scores(scores, epoch, 'DEV')

		dev_classification_report = classification_report(dev_labels, dev_predictions)
		conf = multilabel_confusion_matrix(dev_labels, dev_predictions)
		scores['confusion_matrix'] = conf
		final_scores_per_fold[k] = scores

		with open(os.path.join(write_dir, 'scores', basename+'.csv'), 'w+') as f:
			f.write('\n')
			f.write("FOLD {} \n".format(k))
			f.write('\n')
			f.write("#################################### \n")
			f.write("DEV \n")
			f.write("#################################### \n")
			f.write(dev_classification_report)

	final_scores_per_fold['f1s'] = f1s
	with open(os.path.join(write_dir, 'scores', basename+'.pkl'), 'wb') as f:
		pickle.dump(final_scores_per_fold, f)
	logger.close(best_f1, best_k)
	print("STARTIME {}".format(starttime))
	print("ENDTIME {}".format(time.strftime('%H%M-%b-%d-%Y')))
	print("mean micro f1 {:.2f}".format(np.mean(f1s.values())))