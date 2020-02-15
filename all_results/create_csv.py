
import pickle
import numpy as np
import pandas as pd
import os, sys
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
print(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
"""
process all scores files to create a master CSV files

CNN, BiLSTM and JointBiLSTM
for each emotion there are basically four different model categories where only
thing that changes is body part

amongst scores from each category, find best score and get corresponding epoch.
all results from othe rmodels will be compared to best model and best models best fold!

I want one CSV file with all best results
And I want a pickled file with dataframes

*MODEL* RUN FEAT EMOTION
"""

results_dict = {"model": [], "feats": [],"body_part": [], "interp": [],
				"emotion": [], "shuffle":[], "fold": [], "epoch": [],
				"p":[], "p_std":[], "r":[], "r_std":[], "f":[], "f_std":[],
				"acc":[], "acc_std":[]}


for model_folder in os.scandir('score_files'):
	model = model_folder.name
	for run_subdir in os.scandir(model_folder.path):
		if 'shuffle' in run_subdir.name:
			shuffle = True
		else:
			shuffle = False
		for feature_subdir in os.scandir(run_subdir.path):
			feats = feature_subdir.name
			for emotion_subdir in os.scandir(feature_subdir.path):
				emotion = emotion_subdir.name

				av_scores_this_experiment = []
				all_scores_this_experiment = []

				if model == 'JointBiLSTM':
					scores_subdir = os.path.join(emotion_subdir.path, 'joint/pose/scores')
				else:
					scores_subdir = os.path.join(emotion_subdir.path, 'ind/pose/scores')

				best_f1 = 0
				best_epoch = 0

				working_epoch = []
				epochs_to_add = 0
				for file in os.scandir(scores_subdir):
					working_model = []
					working_feats = []
					working_body_part = []
					working_interp = []
					working_emotion = []
					working_shuffle = []
					working_fold = []

					if file.name.endswith('pkl'):
						print(file.path)
						if 'interp-True' in file.name:
							interp=True
						else:
							interp=False
						if feats == 'stats' or feats == 'no-input':
							body_part = None
						elif "0-hands" in file.name:
							body_part ='hands'
						elif "0-head" in file.name:
							body_part = 'head'
						elif "full-head" in file.name:
							body_part = "full-head"
						elif "full-hh" in file.name:
							body_part = "full-hh"
						elif "full-lr" in file.name:
							body_part = "full"

						with open(file.path, 'rb') as f:
							scores = pickle.load(f)
						av_scores= scores['av_scores']['dev'][1]
						av_acc = scores['av_scores']['dev']['acc']
						av_scores_this_experiment.append(np.concatenate([av_scores, av_acc], axis=1))
						all_scores = scores['all']['dev']
						all_scores_this_experiment.append(all_scores)
						best_current_epoch = np.argmax(av_scores[:,-1])
						best_current_f1 = av_scores[best_current_epoch,-1]
						if best_current_f1 > best_f1:
							best_f1 = best_current_f1
							best_epoch = best_current_epoch

						num_folds = len(all_scores.keys())
						epochs_to_add += (num_folds + 1)
						working_fold = [i for i in range(num_folds)]+['average']
						working_model = [model]*(num_folds+1)
						working_feats = [feats]*(num_folds+1)
						working_body_part = [body_part]*(num_folds+1)
						working_interp = [interp]*(num_folds+1)
						working_emotion = [emotion]*(num_folds+1)
						working_shuffle = [shuffle]*(num_folds+1)

				# for i in range(len(all_scores.keys())):

					results_dict['model'] += working_model
					results_dict['feats'] += working_feats
					results_dict['body_part'] += working_body_part
					results_dict['interp'] += working_interp
					results_dict['emotion'] += working_emotion
					results_dict['shuffle'] += working_shuffle
					results_dict['fold'] += working_fold
				results_dict['epoch'] += [best_epoch] * epochs_to_add
					# results_dict['epoch'].append(best_epoch)

					# print("len av scores this exp", len(av_scores_this_experiment))
					# print("len all scores this exp", len(all_scores_this_experiment))
					# results_dict['model'].append(model)
					# results_dict['feats'].append(feats)
					# results_dict['body_part'].append(body_part)
					# results_dict['interp'].append(interp)
					# results_dict['emotion'].append(emotion)
					# results_dict['shuffle'].append(shuffle)
					# results_dict['fold'].append('average')
					# results_dict['epoch'].append(best_epoch)

				for i, all_scores in enumerate(all_scores_this_experiment):
					scores_per_fold_this_file = []
					for fold in all_scores.keys():
						fold_scores = all_scores[fold][1][best_epoch]
						fold_acc = all_scores[fold]['acc'][best_epoch]
						scores_per_fold_this_file.append(np.concatenate([fold_scores, fold_acc], axis=0))
						results_dict['p'].append(fold_scores[0])
						results_dict['p_std'].append(None)
						results_dict['r'].append(fold_scores[1])
						results_dict['r_std'].append(None)
						results_dict['f'].append(fold_scores[2])
						results_dict['f_std'].append(None)
						results_dict['acc'].append(fold_acc)
						results_dict['acc_std'].append(None)
					scores_per_fold_this_file = np.array(scores_per_fold_this_file)
					# now handle the average
					av_scores = av_scores_this_experiment[i][best_epoch]
					stds = np.std(np.array(scores_per_fold_this_file), axis=0)
					# results_dict['model'].append(model)
					# results_dict['feats'].append(feats)
					# results_dict['body_part'].append(body_part)
					# results_dict['interp'].append(interp)
					# results_dict['emotion'].append(emotion)
					# results_dict['shuffle'].append(shuffle)
					# results_dict['fold'].append('average')
					# results_dict['epoch'].append(best_epoch)
					results_dict['p'].append(av_scores[0])
					results_dict['p_std'].append(stds[0])
					results_dict['r'].append(av_scores[1])
					results_dict['r_std'].append(stds[1])
					results_dict['f'].append(av_scores[2])
					results_dict['f_std'].append(stds[2])
					results_dict['acc'].append(av_scores[3])
					results_dict['acc_std'].append(stds[3])

for key, value in results_dict.items():
	print(key, len(value))
df = pd.DataFrame(results_dict)
df.to_csv('all_results.csv')
