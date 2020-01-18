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
	emotion = args.scores.split('/')[0]
	with open(args.scores, 'rb') as f:
		scores = pickle.load(f)
	train_losses = scores[-1][0]['train']
	dev_losses = scores[-1][0]['dev']
	fig = plt.figure()
	ax = fig.add_subplot(2,1,1)
	ax.plot(train_losses)
	ax.set_title("train loss")

	ax = fig.add_subplot(2,1,2)
	ax.plot(dev_losses)
	ax.set_title("dev loss")
	fig.suptitle(emotion, fontsize=14)
	plt.tight_layout()
	plt.show()