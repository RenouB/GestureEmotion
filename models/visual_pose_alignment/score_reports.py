import os
import sys
import numpy as np
import warnings
import pickle
warnings.filterwarnings('ignore')
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from argparse import ArgumentParser


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('score')
	args = parser.parse_args()

	with open(args.score, "rb") as f:
		scores = pickle.load(f)
	all_dev = scores['all']['dev']

	acc = []
	macros = []
	for i in range(8):
		macros.append(all_dev[i]['macro'])
		acc.append(all_dev[i]['acc'])
	macros = np.array(macros)
	acc = np.array(acc)
	print(macros.mean(axis=0), acc.mean(axis=0))
	print(macros.std(axis=0), acc.std(axis=0))