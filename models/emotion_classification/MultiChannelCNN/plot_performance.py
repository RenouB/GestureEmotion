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
	input = args.scores.split('/')[0]
	body_parts = args.scores.split('/')[-1].split('-')[2:4]
	with open(args.scores, 'rb') as f:
		scores = pickle.load(f)

	# train_losses = []
	# dev_losses =[]
	# for fold in scores['losses'].keys():
	# 	train_losses.append(scores['losses'][fold]['train'])
	# 	dev_losses.append(scores['losses'][fold]['dev'])
	#
	# train_losses = np.array(train_losses).mean(axis=0)
	# dev_losses = np.array(dev_losses).mean(axis=0)
	#
	# fig = plt.figure()
	# ax = fig.add_subplot(2,1,1)
	# ax.plot(train_losses, color="#333cff")
	# ax.plot(dev_losses, color="#0d0f40")
	# ax.set_title("losses")

	# av_scores = scores['av_scores']


	train1_f1s = scores['train'][1][:,-1]
	dev1_f1s = scores['dev'][1][:,-1]
	macro_dev_f1s = scores['dev']['macro'][:,-1]
	# ax = fig.add_subplot(2,1,2)
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.plot(train1_f1s, color="#ff3342", label='train 1')
	ax.plot(dev1_f1s, color="#801a21", label='dev 1')
	ax.plot(macro_dev_f1s, color='blue', label='macro dev')
	ax.set_title("{} {} {} f1s".format(emotion, input, body_parts))
	ax.legend()
	plt.tight_layout()
	plt.savefig(os.path.join('prepare_results/plots',input,emotion, args.scores.split('/')[-1][:-3]+'png'))
