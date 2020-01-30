import os
import sys
import numpy as np
import warnings
import pickle
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from argparse import ArgumentParser

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument('-dir', default='.')
	parser.add_argument('-emotion', default=0, type=int)
	args = parser.parse_args()


	all_scores = {}
	emotions = {0:'anger',1:'happiness',2:'sadness',3:'surprise'}

	best_f1 = 0
	best_epoch = 2000
	best_model = ''
	for root, dirs, files in os.walk(args.dir):
		for file in files:
			if emotions[args.emotion] in root:
				experiment = root.split('/')[1]
				if experiment not in all_scores:
					all_scores[experiment] = {}
				input_dir = root.split('/')[2]
				if input_dir not in all_scores[experiment]:
					all_scores[experiment][input_dir] = {}
				if file.endswith(".pkl"):
					print(os.path.join(root, file))
					file_path = os.path.join(root, file)
					with open(file_path, 'rb') as f:
						scores = pickle.load(f)
					all_scores[experiment][input_dir][file] = scores
					dev_f1s = scores['av_scores']['dev'][1][:,-1]
					best = np.amax(dev_f1s)
					if best > best_f1:
						best_f1 = best
						best_epoch = np.where(dev_f1s == best)[0][0]
						best_model = file
	print('best f1:', best_f1)
	print('best epoch:', best_epoch)
	print('best model:', best_model)
	print('\n')

	for experiment, inputs in all_scores.items():
		print("##################################\n")
		print(experiment)
		for input, files in inputs.items():
			print("###################################\n")
			print(input)
			for file, scores in files.items():
				best_av = scores['av_scores']['dev'][1][best_epoch]
				print(scores['all']['dev'].keys())
				all_folds_best_epoch = np.concatenate([np.expand_dims(scores['all']['dev'][k][1][best_epoch], axis=1) for k in range(8)], axis=1)
				print(file)
				print(np.around(best_av * 100, decimals=2))
				print(np.around(all_folds_best_epoch.std(axis=1) * 100, decimals=2))
				print('\n')
