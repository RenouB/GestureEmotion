import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
np.random.seed(200)
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)

from models.emotion_classification.data.datasets import PoseDataset
from models.evaluation import get_scores, update_scores_per_fold, average_scores_across_folds
import argparse
import logging
from models.pretty_logging import PrettyLogger, get_write_dir
import time

"""
A simple random baseline for emotion classification

"""

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-method', default='random')
	parser.add_argument('-num_folds', default=8, type=int)
	parser.add_argument('-epochs', type=int, default=1)
	parser.add_argument('-joint', action="store_true", default=False)
	parser.add_argument('-modalities', default=0, type=int)
	parser.add_argument("-input", default='no-input')
	parser.add_argument('-attention', default=False)
	parser.add_argument("-interp", default=False)
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
	write_dir = get_write_dir('random', input_type=args.input, joint=args.joint, modalities=args.modalities,
		emotion= args.emotion)
	print(write_dir)
	logger = PrettyLogger(args, os.path.join(write_dir, 'logs'), basename, starttime)

	# TODO: Add different modalities
	data = PoseDataset(args.interval, args.seq_length, args.keypoints,
			args.joint, args.emotion, args.input, args.interp)
	scores_per_fold = {'train':{}, 'dev':{}}

	best_f1 = 0

	for k in range(args.num_folds):
		logger.new_fold(k)
		train_indices, dev_indices = data.split_data(k, args.emotion)
		train_data = Subset(data, train_indices)
		train_loader =DataLoader(train_data, batch_size=len(train_indices), shuffle=True)
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=len(dev_indices))
		train_labels = torch.Tensor([[data[i]['labels']] for i in train_indices])
		dev_labels = torch.Tensor([[data[i]['labels']] for i in dev_indices])

		# get probability of positive class from training data
		prob_1 =  sum(train_labels == 1).item() / len(train_labels)

		if args.method == 'random':
			# use weighted random choice to make predictions
			train_predictions = torch.Tensor([np.random.choice([0,1], size=len(train_labels), p=[1 - prob_1, prob_1])]).reshape(-1,1)
			dev_predictions = torch.Tensor([np.random.choice([0,1], size=len(dev_labels), p=[1 - prob_1, prob_1])]).reshape(-1,1)

		elif args.method == 'majority':
			# always predict majority class
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

		loss = 0
		att_weights = torch.Tensor([[0,0,0]])
		scores_per_fold['train'][k] = {'macro':[], 0:[], 1:[], 'loss':[], 'att_weights':[], 'acc':[]}
		scores_per_fold['dev'][k] = {'macro':[], 0:[], 1:[], 'loss': [], 'att_weights':[], 'acc':[]}
		scores = get_scores(train_labels, train_predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'train',
							loss, att_weights, len(train_data), k)
		logger.update_scores(scores, epoch, 'TRAIN')

		print("################################################")
		print("                     EVAL")
		print("################################################")

		scores = get_scores(dev_labels, dev_predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'dev',
							loss, att_weights, len(dev_data), k)
		logger.update_scores(scores, epoch, 'DEV')

	av_scores = average_scores_across_folds(scores_per_fold)
	scores = {'av_scores':av_scores, 'all':scores_per_fold}
	best_epoch = np.amax(av_scores['dev'][1][:,2]).astype(np.int)
	class_zero_scores = av_scores['dev'][0][best_epoch]
	class_one_scores = av_scores['dev'][1][best_epoch]
	macro_scores = av_scores['dev']['macro'][best_epoch]

	with open(os.path.join(write_dir, 'scores', basename+'.pkl'), 'wb') as f:
		pickle.dump(scores, f)

	with open(os.path.join(write_dir, 'scores', basename+'.csv'), 'a+') as f:
		f.write("BEST EPOCH: {} \n".format(best_epoch))
		f.write("{:>8} {:>8} {:>8} {:>8} {:>8}\n".format("class", "p", "r", "f", "acc"))
		f.write("{:8} {:8.4f} {:8.4f} {:8.4f} \n".format("0", class_zero_scores[0],
			class_zero_scores[1], class_zero_scores[2]))
		f.write("{:8} {:8.4f} {:8.4f} {:8.4f} \n".format("1", class_one_scores[0],
			class_one_scores[1], class_one_scores[2]))
		f.write("{:8} {:8.4f} {:8.4f} {:8.4f} {:8.4f} \n".format("macro",
				macro_scores[0], macro_scores[1], macro_scores[2], av_scores['dev']['acc'][best_epoch][0]))


	logger.close(0, 0)
	print("STARTIME {}".format(starttime))
	print("ENDTIME {}".format(time.strftime('%H%M-%b-%d-%Y')))
