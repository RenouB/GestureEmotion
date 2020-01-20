import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants, cnn_params
from argparse import ArgumentParser


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('scores')
	parser.add_argument('metric')
	args = parser.parse_args()
	emotion = args.scores.split('/')[0]
	with open(args.scores, 'rb') as f:
		scores = pickle.load(f)
	
	if args.metric == 'loss':
		train_losses = []
		dev_losses =[]
		for fold in scores['losses'].keys():
			train_losses.append(scores['losses'][fold]['train'])
			dev_losses.append(scores['losses'][fold]['dev'])
		
		train_losses = np.array(train_losses).mean(axis=0)
		dev_losses = np.array(dev_losses).mean(axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(2,1,1)
		ax.plot(train_losses)
		ax.set_title("train {}".format(args.metric))

		ax = fig.add_subplot(2,1,2)
		ax.plot(dev_losses)
		ax.set_title("dev {}".format(args.metric))
		fig.suptitle(emotion, fontsize=14)
		plt.tight_layout()
		plt.show()

	if args.metric == 'f1':
		train_f1s = []
		dev_f1s =[]
		for fold in scores['f1s'].keys():
			train_f1s.append(scores['f1s'][fold]['train'])
			dev_f1s.append(scores['f1s'][fold]['dev'])
		
		train_f1s = np.array(train_f1s).mean(axis=0)
		dev_f1s = np.array(dev_f1s).mean(axis=0)

		fig = plt.figure()
		ax = fig.add_subplot(2,1,1)
		ax.plot(train_f1s)
		ax.set_title("train {}".format(args.metric))

		ax = fig.add_subplot(2,1,2)
		ax.plot(dev_f1s)
		ax.set_title("dev {}".format(args.metric))
		fig.suptitle(emotion, fontsize=14)
		plt.tight_layout()
		plt.show()