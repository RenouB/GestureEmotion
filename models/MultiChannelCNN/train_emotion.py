import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.nn import functional as F
torch.manual_seed(200)
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from base_models import OneActorOneModalityCNN, TwoActorsOneModalityCNN

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants, cnn_params
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.data.torch_datasets import PoseDataset
from models.evaluation import get_scores
import argparse
import logging
from models.pretty_logging import PrettyLogger, construct_basename, get_write_dir
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

def compute_epoch(model, data_loader, loss_fxn, optim, 
					joint, modalities, print_denominator, train):
	epoch_loss = 0
	epoch_labels = []
	epoch_predictions = []

	batch_counter = 0
	for batch in data_loader:
		batch_counter += 1
		
		if batch_counter % print_denominator == 0:
			print('Processing batch', batch_counter)
		
		# first take care of independent modeling, different modalities
		if not joint:
			if modalities == 0:
				pose = batch['pose']
				labels = batch['labels']
				out = model(pose)		
		# now take care of joint modeling, different modalities
		else:
			if modalities == 0:
				poseA = batch['poseA']
				labelsA = batch['labelsA']
				poseB = batch['poseB']
				labelsB = batch['labelsB']
				out = model(poseA, poseB)
				labels = torch.cat([labelsA, labelsB], dim=0)
		labels = labels.unsqueeze(1)
		predictions = (out > 0.5).int()
		epoch_labels.append(labels)
		epoch_predictions.append(predictions)
		loss = loss_fxn(out, labels.double())
		epoch_loss += loss.item()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

	return torch.cat(epoch_labels, dim=0), torch.cat(epoch_predictions, dim=0), epoch_loss


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-epochs', type=int, default=1)
	parser.add_argument('-joint', action="store_true", default=False)
	parser.add_argument('-modalities', default=0, type=int)
	parser.add_argument('-attention', default=False)
	parser.add_argument('-interval', default=3, type=int)
	parser.add_argument('-seq_length', default=5, type=int)
	parser.add_argument("-emotion", default=0, type=int)
	parser.add_argument("-keypoints", default='full')

	parser.add_argument('-n_filters', default=50, type=int)
	parser.add_argument('-filter_sizes', default=3, type=int)
	parser.add_argument('-cnn_output_dim', default=60, type=int)
	parser.add_argument('-project_output_dim', default=30, type=int)

	parser.add_argument('-batchsize', type=int, default=20)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-l2', type=float, default=0.0001)
	parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')
	parser.add_argument('-optim', default='adam')
	
	parser.add_argument('-cuda', default=False, action='store_true',  help='use cuda')
	parser.add_argument('-gpu', default=0, type=int, help='gpu id')
	
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
	print('epochs', args.epochs)
	if args.cuda and torch.cuda.is_available():
		device = torch.device('cuda:'+str(args.gpu))
	else:
		device = torch.device('cpu')
	print("device: ", device)
	print("batchsize: ", args.batchsize)
	
	# basename for logs, weights
	starttime = time.strftime('%H%M-%b-%d-%Y')
	basename = construct_basename(args)+'-'+starttime
	write_dir = get_write_dir('CNN', attention=False, joint=args.joint, modalities=args.modalities, 
		emotion= args.emotion)
	print(write_dir)
	logger = PrettyLogger(args, os.path.join(write_dir, 'logs'), basename, starttime)

	# TODO: Add different modalities
	data = PoseDataset(args.interval, args.seq_length, args.keypoints, args.joint, args.debug, args.emotion)
	print(data.unique_actor_pairs)
	best_f1 = -1
	best_k = -1
	final_scores_per_fold = {}
	losses = {}
	f1s = {}
	final_f1s = []

	input_dim = get_input_dim(args.keypoints)
	
	for k in range(args.num_folds):
		if not args.joint:
			if args.modalities == 0:
				model = OneActorOneModalityCNN(input_dim, args.n_filters,
							range(1, args.filter_sizes+1), args.cnn_output_dim,
							args.project_output_dim, 1, args.dropout)
		else:
			if args.modalities == 0:
				model = TwoActorsOneModalityCNN(input_dim, args.n_filters, 
							range(1, args.filter_sizes+1), args.cnn_output_dim, 
							args.project_output_dim, 1, args.dropout)

		loss_fxn = torch.nn.BCELoss()
		
		if args.optim == 'adam':
			optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
		elif args.optim == 'sgd':
			optim = SGD(model.parameters(), lr=args.lr)

		model.double()

		losses[k] = {'train':[], 'dev':[]}
		f1s[k] = {'train':[], 'dev':[]}
		logger.new_fold(k)
		train_indices, dev_indices = data.split_data(k, args.emotion)
		train_data = Subset(data, train_indices)
		train_loader =DataLoader(train_data, batch_size=args.batchsize, shuffle=True)
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=args.batchsize)
		print("################################################")
		print("                Beginning fold {}".format(k))
		print("            Length train data: {}".format(len(train_data)))
		print("              Length dev data: {}".format(len(dev_data)))
		print("################################################")
		for epoch in range(args.epochs):
			print("                    TRAIN")
			print("################################################")
			print("                    EPOCH {}".format(epoch))
			print("################################################")
			model.train()
			epoch_labels, epoch_predictions, epoch_loss = compute_epoch(model,
										 train_loader, loss_fxn, optim,
										 args.joint, args.modalities, print_denominator,
										 train=True)

			losses[k]['train'].append(epoch_loss / len(train_data))
			scores = get_scores(epoch_labels, epoch_predictions, detailed=False)
			f1s[k]['train'].append(scores['micro_f'])
			logger.update_scores(scores, epoch, 'TRAIN')

			print("################################################")
			print("                     EVAL")
			print("################################################")
			model.eval()
			dev_labels, dev_predictions, dev_loss = compute_epoch(model,
										 dev_loader, loss_fxn, optim,
										 args.joint, args.modalities, print_denominator,
										 train=False)
			losses[k]['dev'].append(dev_loss / len(dev_data))
			scores = get_scores(dev_labels, dev_predictions, detailed=True)
			f1s[k]['dev'].append(scores['micro_f'])
			logger.update_scores(scores, epoch, 'DEV')



			if scores['micro_f'] > best_f1:
				print('#########################################')
				print('       New best : {:.2f} (previous {:.2f})'.format(scores['micro_f'], best_f1))
				print('         saving model weights')
				print('#########################################')
				best_f1 = scores['micro_f']
				best_k = k
				best_epoch = epoch
				torch.save(model.state_dict(), os.path.join(write_dir, 'weights', basename+'.weights'))
		
		train_classification_report = classification_report(epoch_labels, epoch_predictions)
		dev_classification_report = classification_report(dev_labels, dev_predictions)
		
		conf = multilabel_confusion_matrix(dev_labels, dev_predictions)
		scores['confusion_matrix'] = conf
		final_scores_per_fold[k] = scores
		final_f1s.append(scores['micro_f'])
		with open(os.path.join(write_dir, 'scores', basename+'.csv'), 'a+') as f:
			f.write('\n')
			f.write("FOLD {} \n".format(k))
			f.write("####################################\n")
			f.write("TRAIN \n")
			f.write("####################################\n")
			f.write(train_classification_report)
			f.write('\n')
			f.write("#################################### \n")
			f.write("DEV \n")
			f.write("#################################### \n")
			f.write(dev_classification_report)

	final_scores_per_fold['losses'] = losses
	final_scores_per_fold['f1s'] = f1s
	with open(os.path.join(write_dir, 'scores', basename+'.pkl'), 'wb') as f:
		pickle.dump(final_scores_per_fold, f)
	logger.close(best_f1, best_k)
	print("STARTIME {}".format(starttime))
	print("ENDTIME {}".format(time.strftime('%H%M-%b-%d-%Y')))
	print("best f1: {:.2f}".format(best_f1))
	print("at epoch: {}".format(best_epoch))
	print("last dev loss: {}".format(dev_loss))
	print("mean micro f1 {:.2f}".format(np.mean(final_f1s)))