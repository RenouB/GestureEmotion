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


def average_scores_across_folds(scores_per_fold):
	av_scores = {'train': {}, 'dev':{}}
	for split in ['train', 'dev']:
		for score in ['macro', 0, 1, 'acc']:
			for fold in scores_per_fold[split].keys():
				current_score = scores_per_fold[split][fold][score]
				current_score = np.array(current_score)
				current_score = np.expand_dims(current_score, axis=2)
				if score not in av_scores[split]:
					av_scores[split][score] = current_score
				else:
					av_scores[split][score] = np.concatenate([av_scores[split][score],
						current_score], axis=2)
			av_scores[split][score] = np.mean(av_scores[split][score], axis=2)
	return av_scores

def update_scores_per_fold(scores_per_fold, scores, split, loss, att_weights, data_len, k):
	
	scores_per_fold[split][k]['att_weights'].append(att_weights)
	scores_per_fold[split][k]['loss'].append(loss / data_len)
	scores_per_fold[split][k]['acc'].append(np.expand_dims(np.array(scores['acc']), axis=0))
	scores_per_fold[split][k]['macro'].append(np.array([scores['macro_p'], 
											scores['macro_r'], scores['macro_f']]))
	
	labels = [key for key in scores_per_fold[split][0].keys() if type(key) == int]
	for label in labels:
		try:
			scores_per_fold[split][k][label].append(np.array([scores[label]['p'], scores[label]['r'],
													scores[label]['f']]))
		except:
			scores_per_fold[split][k][label].append(np.array([0,0,0]))
			# scores_per_fold[split][k][1].append(np.array([scores[1]['p'], scores[1]['r'],
			# 									scores[1]['f']]))
	return scores_per_fold
	

def get_scores(labels, predictions):
	labels = labels.cpu().numpy()[:,0]
	predictions = predictions.cpu().numpy()[:,0]
	

	macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(labels, predictions,
									average='macro')
	ps, rs, fs, _ = precision_recall_fscore_support(labels, predictions,
								average=None)
	
	unique_labels = np.unique(labels)
	unique_predictions = np.unique(predictions)

	scores = {}
	scores['macro_p'] = macro_p
	scores['macro_r'] = macro_r
	if np.array_equal(unique_labels, unique_predictions) \
		and len(unique_labels) == 1:
		scores['macro_f'] = fs[unique_labels[0]]
	else:
		scores['macro_f'] = macro_f
	scores['acc'] = sum(labels == predictions) / len(labels)

	for label in unique_labels:
		scores[label] = {}
		scores[label]['p'] = ps[label]
		scores[label]['r'] = rs[label]
		scores[label]['f'] = fs[label]
	return scores
	# classes = np.unique(np.concatenate([labels, predictions], axis=0))
	# scores = {}
	# macro_p = 0
	# macro_r = 0
	
	# for clss in classes:
	# 	scores[clss] = {}
	# 	true_positives = len(np.intersect1d(np.where(labels == clss)[0], 
	# 						np.where(predictions == clss)[0]))
	# 	false_positives = len(np.intersect1d(np.where(labels != clss)[0], 
	# 						np.where(predictions == clss)[0]))
	# 	true_negatives = len(np.intersect1d(np.where(labels != clss)[0], 
	# 						np.where(predictions != clss)[0]))
	# 	false_negatives = len(np.intersect1d(np.where(labels == clss)[0], 
	# 						np.where(predictions != clss)[0]))
	# 	try:
	# 		p = true_positives / (true_positives + false_positives)
	# 	except ZeroDivisionError:
	# 		p = 0
	# 	scores[clss]['p']  = p
		
	# 	try:
	# 		r = true_positives / (true_positives + false_negatives)
	# 	except ZeroDivisionError:
	# 		r = 0
	# 	scores[clss]['r']  = r
		
	# 	try:
	# 		scores[clss]['f'] = 2*p*r / (p+r)
	# 	except ZeroDivisionError:
	# 		scores[clss]['f'] = 0
		
	# 	macro_p += p
	# 	macro_r += r

	# scores['macro_p'] = macro_p / len(classes)
	# scores['macro_r'] = macro_r / len(classes)
	# try:
	# 	scores['macro_f'] = 2*scores['macro_p']*scores['macro_r'] / \
	# 						(scores['macro_p'] + scores['macro_r'])
	# except ZeroDivisionError:
	# 	scores['macro_f'] = 0

	# scores['acc'] = sum(labels == predictions) / len(labels)
	# return scores
