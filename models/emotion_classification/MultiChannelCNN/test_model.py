import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.nn import functional as F
torch.manual_seed(200)
from sklearn.metrics import multilabel_confusion_matrix
from base_models import OneActorOneModalityCNN, TwoActorsOneModalityCNN
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants, cnn_params

from models.data.torch_datasets import PoseDataset
from models.evaluation import get_scores
from train import compute_epoch
from argparse import ArgumentParser

def recreate_params_from_args(args_line):
	args = args_line.split()
	args[0] = args[0].split('(')[1]
	args[-1] = args[-1].split('(')[0]
	as_dct = {}

	for item in args:
		arg, value = item.split('=')
		value = value[:-1]
		if arg == 'attention':
			as_dct[arg] = bool(value)
		elif arg == "batchsize":
			as_dct[arg] = int(value)
		elif arg == 'cnn_output_dim':
			as_dct[arg] = int(value)
		elif arg == 'comment':
			as_dct[arg] = value[1:-1]
		elif arg == 'cuda':
			if value == 'True':
				as_dct[arg] = True
			else:
				as_dct[arg] = False
		elif arg == 'debug':
			if value == 'True':
				as_dct[arg] = True
			else:
				as_dct[arg] = False
		elif arg == 'dropout':
			as_dct[arg] = float(value)
		elif arg == 'epochs':
			as_dct[arg] = int(value)
		elif arg == 'filter_sizes':
			as_dct[arg] = int(value)
		elif arg == 'gpu':
			as_dct[arg] = int(value)
		elif arg == 'interval':
			as_dct[arg] = int(value)
		elif arg == 'joint':
			if value == 'True':
				as_dct[arg] = True
			else:
				as_dct[arg] = False
		elif arg == 'l2':
			as_dct[arg] = float(value)			
		elif arg == 'lr':
			as_dct[arg] = float(value)
		elif arg == 'modalities':
			as_dct[arg] = int(value)
		elif arg == 'n_filters':
			as_dct[arg] = int(value)		
		elif arg == 'num_folds':
			as_dct[arg] = int(value)
		elif arg == 'optim':
			as_dct[arg] = value[1:-1]
		elif arg == 'seq_length':
			as_dct[arg] = int(value)
		elif arg == 'test':
			if value == 'True':
				as_dct[arg] = True
			else:
				as_dct[arg] = False
	return as_dct

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-logfile')
	arguments = parser.parse_args()

	with open(arguments.logfile, 'r') as f:
		logfile = f.read().split('\n')
	
	for line in logfile:
		if line.startswith('BEST_SCORE'):
			best_f1 = float(line.split()[-1])
		if line.startswith('AT K'):
			best_k = int(line.split()[-1])

	args_line = logfile[0]
	args = recreate_params_from_args(args_line)
	print(args_line)
	print(args)

	weights_path = []
	for item in arguments.logfile.split('/'):
		if item == 'logs':
			weights_path.append('weights')
		else:
			weights_path.append(item)	
	weights_path[-1] = weights_path[-1][:-4]+'.weights'
	weights_path = "/".join(weights_path)

	if not args["joint"]:
		if args["modalities"] == 0:
			model = OneActorOneModalityCNN(cnn_params["POSE_DIM"], args["n_filters"],
						range(1, args["filter_sizes"]+1), args["cnn_output_dim"],
						cnn_params["NUM_CLASSES"], args["dropout"])
	else:
		if args["modalities"] == 0:
			model = TwoActorsOneModalityCNN(cnn_params["POSE_DIM"], args["n_filters"],
						range(1, args["filter_sizes"]+1), args["cnn_output_dim"],
						cnn_params["NUM_CLASSES"], args["dropout"])
	
	model.load_state_dict(torch.load(weights_path))
	model.double()
	model.eval()
	loss_fxn = torch.nn.BCEWithLogitsLoss()
	
	data = PoseDataset(args['interval'], args['seq_length'], args['joint'], args['debug'])
	train_indices, dev_indices = data.split_data(best_k)
	dev_data = Subset(data, dev_indices)
	dev_loader = DataLoader(dev_data, args['batchsize'])
	
	labels, predictions, loss = compute_epoch(model, dev_loader, loss_fxn, optim=None, train=False,
								joint=args['joint'], modalities=args['modalities'], 
								print_denominator=20)

	scores = get_scores(labels, predictions, detailed=False)
	print("BEST F1 FROM LOGFILE: {:.2f}".format(best_f1))
	best_f1 = scores['micro_f']
	print("BEST F1 FROM LOADED MODEL: {:.2f}".format(best_f1))


	