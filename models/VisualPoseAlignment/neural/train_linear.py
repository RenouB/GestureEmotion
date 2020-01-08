import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

torch.manual_seed(200)
from sklearn.metrics import multilabel_confusion_matrix, precision_recall_fscore_support


PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])

sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)

import argparse
import logging
from models.pretty_logging import PrettyLogger, construct_basename, get_logs_weights_scores_dirs
import time

HISTOGRAMS_DATA_DIR = constants["HISTOGRAMS_DATA_DIR"]

def get_scores(labels, predictions, detailed):
	micro_p, micro_r, micro_f, _ = precision_recall_fscore_support(labels, predictions,
			 average='micro')
	macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(labels, predictions, 
			average='macro')
	prfs_per_class = precision_recall_fscore_support(labels, predictions, average=None)
	
	overall_acc = sum(labels == predictions).item() / len(predictions)
	
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

class TrainDataset(Dataset):
	def __init__(self, actor_pair):
		self.X = actor_pair['hists'].squeeze(3)
		self.X = self.X.reshape(-1,self.X.shape[2])
		self.labels = actor_pair['labels'].reshape(-1)


	def __len__(self):
		return len(self.X) - 1

	def __getitem__(self, i):
		return {'hist':self.X[i], 'label':self.labels[i]}

class TestDataset(Dataset):
	def __init__(self, actor_pair):
		self.X = actor_pair['hists'].squeeze(3)
		self.labels = actor_pair['labels']

	def __len__(self):
		return len(self.X) - 1

	def __getitem__(self, i):
		return {'hist':self.X[i], 'label':self.labels[i]}

def compute_train_epoch(model, data_loader, loss_fxn, optim, 
					print_denominator):
	epoch_loss = 0
	epoch_labels = []
	epoch_predictions = []

	batch_counter = 0
	for batch in data_loader:
		batch_counter += 1
		
		if batch_counter % print_denominator == 0:
			print('Processing batch', batch_counter)
		hists = batch['hist']
		hists = hists.double()
		labels = batch['label'].long()
		out = model(hists)
		predictions = out.argmax(axis=1).long()
		epoch_labels.append(labels)
		epoch_predictions.append(predictions)

		loss = loss_fxn(out, labels)
		epoch_loss += loss.item()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

	return torch.cat(epoch_labels, dim=0), torch.cat(epoch_predictions, dim=0), epoch_loss

def compute_test_epoch(model, data_loader, loss_fxn, optim, 
				print_denominator):
	epoch_loss = 0
	epoch_labels = []
	epoch_predictions = []

	batch_counter = 0
	for batch in data_loader:
		batch_counter += 1

		if batch_counter % print_denominator == 0:
			print('Processing batch', batch_counter)
		
		hists = batch['hist']
		hists = hists.double()
		epoch_labels.append(batch['label'][0])	
		out = model(hists).squeeze(0)
		predictions = out.argmax(1)
		
		num_zeroes = sum(predictions == 0)
		num_ones = sum(predictions == 1)
		
		if num_zeroes > num_ones:
			epoch_predictions += [1] * len(out)
		else:
			epoch_predictions += [0] * len(out)
	
	return torch.cat(epoch_labels, dim=0), torch.Tensor(epoch_predictions), epoch_loss	

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-color', default='hsv')
	parser.add_argument('-num_bins', default=32, type=int)
	parser.add_argument('-only_hue', action="store_true", default=False)

	parser.add_argument('-epochs', type=int, default=1)
	parser.add_argument('-joint', action="store_true", default=False)

	parser.add_argument('-batchsize', type=int, default=20)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-l2', type=float, default=0.0001)
	parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')
	parser.add_argument('-optim', default='adam')
	
	parser.add_argument('-cuda', default=False, action='store_true',  help='use cuda')
	parser.add_argument('-gpu', default=0, type=int, help='gpu id')
	
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
	basename = '-'.join([args.color, 'only_hue', str(args.only_hue), str(args.num_bins)])+'-'
	log_dir = './outputs/logs'
	weights_dir = './outputs/weights'
	logger = PrettyLogger(args, log_dir, basename, starttime)

	# TODO: Add different modalities
	
	loss_fxn = torch.nn.CrossEntropyLoss()
	

	with open(os.path.join(HISTOGRAMS_DATA_DIR, basename+'train.pkl'), 'rb') as f:
		train = pickle.load(f)
	with open(os.path.join(HISTOGRAMS_DATA_DIR, basename+'test.pkl'), 'rb') as f:
		test = pickle.load(f)
	print(train.keys())
	
	input_dim = train['0102']['hists'].shape[2]
	model = torch.nn.Sequential(nn.Dropout(args.dropout), nn.Linear(input_dim, 50), nn.ReLU(), 
		 							nn.Dropout(args.dropout), nn.Linear(50, 2))
	if args.optim == 'adam':
		optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
	
	elif args.optim == 'sgd':
		optim = SGD(model.parameters(), lr=args.lr)

	model.double()
	

	for pair_id in ['0708', '1314']:	
		train_data = TrainDataset(train[pair_id])
		train_loader = DataLoader(train_data, batch_size = args.batchsize)
		test_data = TestDataset(test[pair_id])
		test_loader = DataLoader(test_data, batch_size = 1)
		
		# print(len(train[pair_id]))
		logging.info('\n')
		logging.info("################################################")
		logging.info("            PROCESSING PAIR{}".format(pair_id))
		logging.info("################################################")
		for epoch in range(args.epochs):
			print("################################################")
			print("                    TRAIN")
			print("################################################")
			print("                    EPOCH {}".format(epoch))
			print("################################################")
			model.train()
			epoch_labels, epoch_predictions, epoch_loss = compute_train_epoch(model,
										 train_loader, loss_fxn, optim, print_denominator)
			print(epoch_labels.shape)
			print('//////////////')
			print(epoch_predictions.shape)
			scores = get_scores(epoch_labels, epoch_predictions, detailed=False)
			logger.update_scores(scores, epoch, 'TRAIN')

			print("################################################")
			print("                     EVAL")
			print("################################################")
			model.eval()
			test_labels, test_predictions, test_loss = compute_test_epoch(model,
										 test_loader, loss_fxn, optim, print_denominator)
			scores = get_scores(test_labels, test_predictions, detailed=True)
			logger.update_scores(scores, epoch, 'DEV')


			# if scores['micro_f'] > best_f1:
			# 	print('#########################################')
			# 	print('       New best : {:.2f} (previous {:.2f})'.format(scores['micro_f'], best_f1))
			# 	print('         saving model weights')
			# 	print('#########################################')
			# 	best_f1 = scores['micro_f']
			# 	best_k = k
			# 	torch.save(model.state_dict(), os.path.join(weights_dir, basename+'.weights'))
		
		# conf = multilabel_confusion_matrix(dev_labels, dev_predictions)
		# scores['confusion_matrix'] = conf

	# with open(os.path.join(scores_dir, basename+'.pkl'), 'wb') as f:
	# 	pickle.dump(final_scores_per_fold, f)
	# logger.close(best_f1, best_k)
	print("STARTIME {}".format(starttime))
	print("ENDTIME {}".format(time.strftime('%H%M-%b-%d-%Y')))
	print("best f1: {:.2f}".format(best_f1))