import os
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants, cnn_params
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)

for keypoints in ['full', 'full-hh', 'full-head', 'head', 'hands', 'random', 'majority']:
	print('\n##########################################################')
	print(keypoints.upper())
	print('##########################################################\n')

	for split in ['train', 'dev']:
		print(split.upper())
		print('--------------------------------------------------------------')
		print('{:>10} {:>10} {:>10} {:>10} {:>10}'.format('','p','r','f','acc'))
		all_emotion_means = []
		all_emotion_stds = []
		for emotion in ['anger', 'happiness', 'sadness', 'surprise']:
			means_to_print = '{:>10} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
			stds_to_print = '{:>10} {:>10.2f} {:>10.2f} {:>10.2f} {:>10.2f}'
			if keypoints in ['random', 'majority']:
				emotion_scores_dir = os.path.join(MODELS_DIR, 'rand', emotion, 'ind/pose/scores')	
			else:
				emotion_scores_dir = os.path.join(MODELS_DIR, 'MultiChannelCNN', emotion, 'ind/pose/scores')
			
			files = [file for file in os.listdir(emotion_scores_dir) if file.endswith(".csv") and 'ep60' in file]
			
			for file in files:
				if keypoints in ['random', 'majority']:
					file_keypoints = file.split('-')[0]
				else:
					file_keypoints = '-'.join(file.split('-')[2:4])
				if keypoints == file_keypoints or (keypoints+'-NO' == file_keypoints):
					with open(os.path.join(emotion_scores_dir, file)) as f:
						lines = f.read().split('\n')
						weighted_av_lines = [line for line in lines if line.startswith('   macro')]
						
						scores = [line.split()[2:-1] for line in weighted_av_lines]
						scores = [[float(score) for score in line] for line in scores]
						acc_lines = [line for line in lines if 'accuracy' in line]
						accs = [float(line.split()[-2]) for line in acc_lines]
						scores = np.array(scores)

						train_scores = scores[[[i for i in range(0, scores.shape[0], 2)]]]
						dev_scores = scores[[[i for i in range(1,scores.shape[0], 2)]]]
									
						accs = np.array(accs)
						# print(accs)
						
						train_accs = accs[[i for i in range(0, accs.shape[0], 2)]]
						dev_accs = accs[[i for i in range(1,accs.shape[0], 2)]]
						
						train_scores = np.concatenate([train_scores, np.expand_dims(train_accs, 1)], axis=1)
						train_stds = np.std(train_scores, axis=0)
						train_means = np.mean(train_scores, axis=0)

						dev_scores = np.concatenate([dev_scores, np.expand_dims(dev_accs, 1)], axis=1)
						dev_stds = np.std(dev_scores, axis=0)
						dev_means = np.mean(dev_scores, axis=0)
					

					if split == 'train':
						all_emotion_means.append(train_means)
						all_emotion_stds.append(train_stds)
						print(means_to_print.format(emotion, train_means[0], train_means[1], train_means[2], train_means[3]))
						print(stds_to_print.format('stds', train_stds[0], train_stds[1], train_stds[2], train_stds[3]))

					else:
						all_emotion_means.append(dev_means)
						all_emotion_stds.append(dev_stds)
						print(means_to_print.format(emotion, dev_means[0], dev_means[1], dev_means[2], dev_means[3]))
						print(stds_to_print.format('stds', dev_stds[0], dev_stds[1], dev_stds[2], dev_stds[3]))
		
		if not len(all_emotion_means) or not len(all_emotion_stds):
			continue
		all_emotion_means = np.concatenate([np.expand_dims(row, 0) for row in all_emotion_means], axis=0)
		all_emotion_stds = np.concatenate([np.expand_dims(row, 0) for row in all_emotion_stds], axis=0)
		all_emotion_means = np.mean(all_emotion_means, axis=0)
		all_emotion_stds = np.mean(all_emotion_stds, axis=0)
		print(means_to_print.format('all', all_emotion_means[0], all_emotion_means[1],
					 all_emotion_means[2], all_emotion_means[3]))
		print(stds_to_print.format('all', all_emotion_stds[0], all_emotion_stds[1],
					 all_emotion_stds[2], all_emotion_stds[3]))
		print('\n')	


