import os
import sys
import logging
import time
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]


def get_scores(labels, predictions, detailed):
	micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(labels, predictions, average='micro')
	macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(labels, predictions, average='macro')
	prfs_per_class = precision_recall_fscore_support(labels, predictions, average=None)

	exact_acc = len([i for i, pred in enumerate(predictions) \
					 if torch.all(torch.eq(predictions[i], labels[i]))]) \
					  / len(labels)

	acc_per_class = np.array([sum(labels[:,i] == predictions[:,i]) / \
						len(labels) for i in range(len(labels[0]))])
	av_acc = acc_per_class.mean()

	scores = {'exact_acc': exact_acc, 'av_acc': av_acc,
			'micro_p':micro_p, 'micro_r':micro_r, 'micro_f':micro_f,
			'macro_p': macro_p, 'macro_r': macro_r, 'macro_f':macro_f}

	if detailed:
		for i in range(len(labels[0])):
			scores[i] = {}
			scores[i]['acc'] = acc_per_class[i]
			scores[i]['p'] = prfs_per_class[0][i]
			scores[i]['r'] = prfs_per_class[1][i]
			scores[i]['f'] = prfs_per_class[2][i]

	return scores
