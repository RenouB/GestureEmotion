import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch.optim import Adam, SGD
from torch.nn import functional as F
from joint_bilstm import JointBiLSTM
torch.backends.cudnn.deterministic = True

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
from models.pretty_logging import PrettyLogger, construct_basename, get_write_dir
import time

"""
train and evaluate a JointBiLSTM for emotion classification
this script is not systematically commented. it is analogous to
BiLSTM/train_bilstm.py, which is commented.
"""

def get_input_dim(keypoints, input):
	"""
	get input dimension of keypoints for a given body part subset
	keypoints: full, full-hh, full-head, head or hands
	input: deltas-noatt, deltas or brute
	"""
	if keypoints == 'full':
		dim = len(constants["WAIST_UP_BODY_PART_INDICES"]) * 2
	if keypoints == 'full-hh':
		dim = len(constants["FULL-HH"]) * 2
	if keypoints == 'full-head':
		dim = len(constants["FULL-HEAD"]) * 2
	if keypoints == 'head':
		dim = len(constants["HEAD"]) * 2
	if keypoints == 'hands':
		dim = len(constants["HANDS"]) * 2

	if input == 'deltas-noatt':
		return dim * 3

	return dim

def compute_epoch(model, data_loader, loss_fxn, optim,
					joint, modalities, print_denominator, train):
	"""
	compute one training/dev epoch
	model: neural network
	data_loader: train or dev data loader
	loss_fxn:
	optim:
	joint: deprecated param
	modalities: deprecated param
	print_denominator: print more often for debugging
	train: bool. indicates whether train or dev
	"""
	epoch_loss = 0
	epoch_labels = []
	epoch_predictions = []
	epoch_att_weights = []
	batch_counter = 0
	for batch in data_loader:
		batch_counter += 1

		if batch_counter % print_denominator == 0:
			print('Processing batch', batch_counter)

		# first take care of independent modeling, different modalities

		labelsA = batch['labelsA']
		labelsB = batch['labelsB']

		if args.input == 'brute':
			poseA = batch['poseA']
			poseB = batch['poseB']
			# att weights are empty placeholder

		elif args.input == 'deltas-noatt':
			poseA = torch.cat([batch['poseA'][:,-3:,:], batch['deltasA'][:,-3:,:],
					batch['delta_deltasA']], axis=2)
			poseB = torch.cat([batch['poseB'][:,-3:,:], batch['deltasB'][:,-3:,:],
					batch['delta_deltasB']], axis=2)

		outA, outB, att_weightsA, att_weightsB = model(poseA, poseB)

		labelsA = labelsA.unsqueeze(1)
		labelsB = labelsB.unsqueeze(1)
		predictionsA = (outA > 0.5).int()
		predictionsB = (outB > 0.5).int()
		epoch_labels.append(labelsA)
		epoch_labels.append(labelsB)
		epoch_predictions.append(predictionsA)
		epoch_predictions.append(predictionsB)
		epoch_att_weights.append(att_weightsA)
		epoch_att_weights.append(att_weightsB)

		loss = loss_fxn(torch.cat([outA, outB], dim=0),
			torch.cat([labelsA.double(), labelsB.double()], dim=0))
		epoch_loss += loss.item()

		if train:
			optim.zero_grad()
			loss.backward()
			optim.step()

	return torch.cat(epoch_labels, dim=0), torch.cat(epoch_predictions, dim=0), \
			epoch_loss, torch.cat(epoch_att_weights, axis=0)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-epochs', type=int, default=1, help="number of epochs")
	parser.add_argument('-joint', action="store_true", default=True,
						help="if joint, models emotions of actors jointly")
	parser.add_argument('-modalities', default=0, type=int, help="deprecated")
	parser.add_argument('-interval', default=3, type=int, help="deprecated. keep default")
	parser.add_argument('-seq_length', default=5, type=int, help="deprecated. keep default")
	parser.add_argument('-interp', action='store_true', default=True,
						help="deprecated. keep default. use interpolated data.")
	parser.add_argument("-emotion", default=0, type=int,
						help="anger: 0, happiness: 1, sadness: 2, surprise: 3")
	parser.add_argument("-keypoints", default='full',
						help="full, full-head, full-hh, head or hands")
	parser.add_argument("-input", default="brute", help="brute or deltas-noatt")
	parser.add_argument("-hidden_size", default=60, help="lstm hidden state size")
	parser.add_argument("-attention_dim", default=60, type=int,
						help="dimensionality of evidence layer outputs and att vector")
	parser.add_argument('-lstm_layers', default=2, type=int, help="num lstm layers")
	parser.add_argument('-batchsize', type=int, default=20)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-l2', type=float, default=0.001)
	parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')
	parser.add_argument('-optim', default='adam', help="deprecated. keep default")
	parser.add_argument('-cuda', default=False, action='store_true',  help='use cuda')
	parser.add_argument('-num_folds', default=8, type=int,
						help="reducing number of folds will exclude certain actor pairs")
	parser.add_argument('-test', action='store_true', default=False, help="deprecated")
	parser.add_argument('-debug', action='store_true', default=False,
						help="use very small subset of data, for debugging")
	parser.add_argument('-comment', default='', help="you may append a comment to all outputs")
	args = parser.parse_args()


	if args.debug:
		print_denominator = 5
	else:
		print_denominator = 100

	print("################################################")
	print("                  STARTING")
	print('epochs', args.epochs)
	if args.cuda:
		use_gpu = lambda x=True: torch.set_default_tensor_type(torch.cuda.FloatTensor
											 if torch.cuda.is_available() and x
											 else torch.FloatTensor)
		use_gpu()
		device = torch.cuda.current_device()
	else:
		device = torch.device('cpu')
	print("device: ", device)
	print("batchsize: ", args.batchsize)

	# basename for logs, weights
	starttime = time.strftime('%H%M-%b-%d-%Y')
	basename = construct_basename(args)+'-'+starttime
	write_dir = get_write_dir('JointBiLSTM', input_type = args.input, joint=args.joint,
					modalities=args.modalities, emotion= args.emotion)
	print(write_dir)
	logger = PrettyLogger(args, os.path.join(write_dir, 'logs'), basename, starttime)

	# TODO: Add different modalities
	data = PoseDataset(args.interval, args.seq_length, args.keypoints,
			args.joint, args.emotion, args.input, args.interp)
	input_dim = get_input_dim(args.keypoints, args.input)
	scores_per_fold = {'train':{}, 'dev':{}}


	for k in range(args.num_folds):
		best_f1 = 0

		logger.new_fold(k)

		torch.manual_seed(200)
		torch.cuda.manual_seed(200)
		model = JointBiLSTM(input_dim, args.hidden_size, args.attention_dim,
					args.lstm_layers, args.dropout)

		loss_fxn = torch.nn.BCELoss()

		if args.optim == 'adam':
			optim = Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
		elif args.optim == 'sgd':
			optim = SGD(model.parameters(), lr=args.lr)

		model.double()

		train_indices, dev_indices = data.split_data(k, args.emotion)
		if args.debug:
			train_indices = train_indices[:1500]
			dev_indices = dev_indices[:1500]
		train_data = Subset(data, train_indices)
		dev_data = Subset(data, dev_indices)
		dev_loader = DataLoader(dev_data, batch_size=args.batchsize)
		print("################################################")
		print("                Beginning fold {}".format(k))
		print("            Length train data: {}".format(len(train_data)))
		print("              Length dev data: {}".format(len(dev_data)))
		print("################################################")


		scores_per_fold['train'][k] = {'macro':[], 0:[], 1:[], 'loss':[], 'att_weights':[], 'acc':[]}
		scores_per_fold['dev'][k] = {'macro':[], 0:[], 1:[], 'loss': [], 'att_weights':[], 'acc':[]}

		for epoch in range(args.epochs):
			torch.manual_seed(epoch)
			torch.cuda.manual_seed(epoch)
			train_loader =DataLoader(train_data, batch_size=args.batchsize, shuffle=True, num_workers=0)
			print("                    TRAIN")
			print("################################################")
			print("                    EPOCH {}".format(epoch))
			print("################################################")
			model.train()
			epoch_labels, epoch_predictions, epoch_loss, epoch_att_weights = \
										compute_epoch(model,
										 train_loader, loss_fxn, optim,
										 args.joint, args.modalities, print_denominator,
										 train=True)
			epoch_labels = epoch_labels.cpu()
			epoch_predictions = epoch_predictions.cpu()

			epoch_att_weights = epoch_att_weights.cpu().numpy()
			scores = get_scores(epoch_labels, epoch_predictions)
			scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'train',
								epoch_loss, epoch_att_weights, len(train_data), k)
			logger.update_scores(scores, epoch, 'TRAIN')


			print("################################################")
			print("                     EVAL")
			print("################################################")
			model.eval()
			dev_labels, dev_predictions, dev_loss , dev_att_weights = \
										compute_epoch(model,
										 dev_loader, loss_fxn, optim,
										 args.joint, args.modalities, print_denominator,
										 train=False)
			dev_labels = dev_labels.cpu()
			dev_predictions = dev_predictions.cpu()

			dev_att_weights = dev_att_weights.cpu().numpy()
			scores = get_scores(dev_labels, dev_predictions)
			scores_per_fold = update_scores_per_fold(scores_per_fold, scores, 'dev',
								dev_loss, dev_att_weights, len(dev_data), k)
			logger.update_scores(scores, epoch, 'DEV')


		f1 = scores[0]['f']
		if f1 > best_f1:
			print('#########################################')
			print('       New best : {:.2f} (previous {:.2f})'.format(f1, best_f1))
			print('        at fold : {}'.format(k))
			print('         saving model weights')
			print('#########################################')
			best_f1 = f1
			best_epoch = epoch
			best_fold = k
			torch.save(model.state_dict(), os.path.join(write_dir, 'weights', basename+'fold{}'.format(k)+'.weights'))



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
