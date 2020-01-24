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
	args = parser.parse_args()
	emotion = args.scores.split('/')[1]
	with open(args.scores, 'rb') as f:
		scores = pickle.load(f)
	
	train_losses = []
	dev_losses =[]
	for fold in scores['losses'].keys():
		train_losses.append(scores['losses'][fold]['train'])
		dev_losses.append(scores['losses'][fold]['dev'])
	
	train_losses = np.array(train_losses).mean(axis=0)
	dev_losses = np.array(dev_losses).mean(axis=0)

	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)
	ax.plot(train_losses, color="#333cff")
	ax.plot(dev_losses, color="#0d0f40")
	ax.set_title("losses")

	train_f1s = []
	dev_f1s =[]
	for fold in scores['f1s'].keys():
		train_f1s.append(scores['f1s'][fold]['train'])
		dev_f1s.append(scores['f1s'][fold]['dev'])
	
	train_f1s = np.array(train_f1s).mean(axis=0)
	dev_f1s = np.array(dev_f1s).mean(axis=0)

	ax = fig.add_subplot(2,1,2)
	ax.plot(train_f1s, color="#ff3342")
	ax.plot(dev_f1s, color="#801a21")
	ax.set_title("f1s")
	ax.legend()
	fig.suptitle(emotion, fontsize=14)
	plt.tight_layout()
	plt.show()