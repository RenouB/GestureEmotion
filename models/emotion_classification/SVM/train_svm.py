import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.svm import SVC
np.random.seed(200)


PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.emotion_classification.data.datasets import SvmPoseDataset
from models.evaluation import get_scores, update_scores_per_fold, average_scores_across_folds
import argparse
import logging
from models.pretty_logging import PrettyLogger, construct_basename, get_write_dir
import time

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-emotion", default=0, type=int)
	parser.add_argument('-C', type=float, default=1)
	parser.add_argument('-kernel', default='rbf')
	parser.add_argument('-class_weight', action="store_true", default=False)
	parser.add_argument('-num_folds', default=8, type=int)
	parser.add_argument('-test', action='store_true', default=False)
	parser.add_argument('-debug', action='store_true', default=False)
	parser.add_argument('-interp', action='store_true', default=False)
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
	basename = '-'.join(['svm', 'c-'+str(args.C),'interp-'+str(args.interp), args.kernel, str(args.C), 'cw', str(args.class_weight)])
	write_dir = get_write_dir('SVM', joint=False, input_type='',
	 				modalities=0, emotion= args.emotion)
	print(write_dir)
	logger = PrettyLogger(args, os.path.join(write_dir, 'logs'), basename, starttime)

	# TODO: Add different modalities
	data = SvmPoseDataset(args.emotion, args.interp)

	scores_per_fold = {'train':{}, 'dev':{}}
	best_f1 = 0
	epoch = 0
	for k in range(args.num_folds):
		logger.new_fold(k)
		scores_per_fold['train'][k] = {'macro':[], 0:[], 1:[], 'loss':[], 'att_weights':[], 'acc':[]}
		scores_per_fold['dev'][k] = {'macro':[], 0:[], 1:[], 'loss': [], 'att_weights':[], 'acc':[]}

		train_indices, dev_indices = data.split_data(k)

		if args.debug:
			train_indices = train_indices[:9000]
			dev_indices = dev_indices[:9000]

		train_data = Subset(data, train_indices)
		train_loader = DataLoader(train_data, batch_size=len(train_indices))
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=len(dev_indices))

		print("################################################")
		print("                Beginning fold {}".format(k))
		print("            Length train data: {}".format(len(train_data)))
		print("              Length dev data: {}".format(len(dev_data)))
		print("################################################")

		if args.class_weight:
			class_weight = balanced
			svm = SVC(C=args.C, kernel=args.kernel, class_weight=class_weight,
					random_state=200)
		else:
			svm = SVC(C=args.C, kernel=args.kernel, random_state=200)

		train = next(iter(train_loader))
		print("Beginning training")
		train_X = train['features'].numpy()
		print(train_X.shape)
		train_y = train['label'].numpy()
		print(train_y.shape)
		svm.fit(train_X, train_y)
		train_predictions = svm.predict(train_X)
		scores = get_scores(train_y, train_predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'train',
		0, [0,0,0], len(train_data), k)
		logger.update_scores(scores, 0, 'TRAIN')
		print("Beginning inference")
		dev = next(iter(dev_loader))
		dev_X = dev['features'].numpy()
		dev_y = dev['label'].numpy()
		dev_predictions = svm.predict(dev_X)

		scores = get_scores(dev_y, dev_predictions)
		scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'dev',
							0, [0,0,0], len(dev_data), k)
		logger.update_scores(scores, 0, 'DEV')

		unique_labels = np.unique(dev_y)
		print(unique_labels)
		if 1 in unique_labels:
			print()
			f1 = scores['macro_f']
			print(scores)
			print(f1)
			if f1 > best_f1:
				best_f1 = f1
				best_epoch = epoch
				best_fold = k

	av_scores = average_scores_across_folds(scores_per_fold)
	scores = {'av_scores': av_scores, 'all': scores_per_fold}

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
