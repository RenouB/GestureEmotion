import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-method', default='random')
	parser.add_argument('-num_folds', default=8, type=int)
	parser.add_argument('-epochs', type=int, default=1)
	parser.add_argument('-joint', action="store_true", default=False)
	parser.add_argument('-modalities', default=0, type=int)
	parser.add_argument('-attention', default=False)
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_length', default=5, type=int)
	parser.add_argument("-emotion", default=0, type=int)
	parser.add_argument("-keypoints", default='full')
	parser.add_argument('-test', action='store_true', default=False)
	parser.add_argument('-debug', action='store_true', default=False)
	parser.add_argument('-comment', default='')
	args = parser.parse_args()
	
	epoch = 0	
	print("################################################")
	print("                  STARTING")
	
	# basename for logs, weights
	starttime = time.strftime('%H%M-%b-%d-%Y')
	basename = '-'.join([args.method, str(args.interval), str(args.seq_length)])
	write_dir = get_write_dir('random', attention=False, joint=args.joint, modalities=args.modalities, 
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
	final_f1s = []
	
	for k in range(args.num_folds):
		f1s[k] = {'train':[], 'dev':[]}
		logger.new_fold(k)
		train_indices, dev_indices = data.split_data(k, args.emotion)
		train_data = Subset(data, train_indices)
		train_loader =DataLoader(train_data, batch_size=len(train_indices), shuffle=True)
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=len(dev_indices))
		train_labels = torch.Tensor([[data[i]['labels']] for i in train_indices])
		dev_labels = torch.Tensor([[data[i]['labels']] for i in dev_indices])

		prob_1 =  len(np.where(train_labels == 1)) / len(train_labels)
		
		if args.method == 'random':
			train_predictions = torch.Tensor([np.random.choice([0,1], size=len(train_labels), p=[1 - prob_1, prob_1])]).reshape(-1,1)
			
			dev_predictions = torch.Tensor([np.random.choice([0,1], size=len(dev_labels), p=[1 - prob_1, prob_1])]).reshape(-1,1)
			
		elif args.method == 'majority':
			if prob_1 >= 0.5:
				train_predictions = torch.Tensor([1]*len(train_labels)).unsqueeze(1)
				dev_predictions = torch.Tensor([1]*len(dev_labels)).unsqueeze(1)
			else:
				train_predictions = torch.Tensor([0]*len(train_labels)).unsqueeze(1)
				dev_predictions = torch.Tensor([0]*len(dev_labels)).unsqueeze(1)

		print("################################################")
		print("                Beginning fold {}".format(k))
		print("            Length train data: {}".format(len(train_data)))
		print("              Length dev data: {}".format(len(dev_data)))
		print("################################################")
		print("                    TRAIN")
		print("################################################")
		
		
		scores = get_scores(train_labels, train_predictions, detailed=False)
		f1s[k]['train'].append(scores['micro_f'])
		
		logger.update_scores(scores, epoch, 'TRAIN')

		train_classification_report = classification_report(train_labels, train_predictions)
		conf = multilabel_confusion_matrix(train_labels, train_predictions)
		scores['confusion_matrix'] = conf

		
		print("################################################")
		print("                     EVAL")
		print("################################################")
		
		scores = get_scores(dev_labels, dev_predictions, detailed=False)
		f1s[k]['dev'].append(scores['micro_f'])
		logger.update_scores(scores, epoch, 'DEV')

		dev_classification_report = classification_report(dev_labels, dev_predictions)
		conf = multilabel_confusion_matrix(dev_labels, dev_predictions)
		scores['confusion_matrix'] = conf
		final_scores_per_fold[k] = scores
		final_f1s.append(scores['micro_f'])

		with open(os.path.join(write_dir, 'scores', basename+'.csv'), 'a+') as f:
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
	print("mean micro f1 {:.2f}".format(np.mean(final_f1s)))