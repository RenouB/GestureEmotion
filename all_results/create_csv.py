
import pickle
import numpy as np
import pandas as pd
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
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
                "emotion": [], "fold": [], "epoch": [],
                "p":[], "p_std":[], "r":[], "r_std":[], "f":[], "f_std":[],
                "acc":[], "acc_std":[]}


for model_folder in os.scandir('.'):
    model = model_folder.name
	for run_subdir in os.scandir(model_folder.path):
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
            for file in os.scandir(scores_subdir):
                if file.name.endswith('pkl'):
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
                            bdoy_part = "full"

                    with open(file.path, 'rb') as f:
                        scores = pickle.load(f)
                    av_scores= scores['av_scores']['dev'][1]
                    av_acc = scores['av_scores']['dev']['acc']
                    av_scores_this_experiment.append(np.concatenate([av_scores, av_acc], axis=1))
                    all_scores = scores['all']['dev']
                    all_scores_this_experiment.append(all_scores)
                    best_current_epoch = np.argmax(all_scores[:,-1])
                    best_current_f1 = av_scores[best_current_epoch,-1]
                    if best_current_f1 > best_f1:
                        best_f1 = best_current_f1
                        best_epoch = best_current_epoch

                    for fold in all_scores.keys():
                        results_dict['model'].append(model)
                        results_dict['feats'].append(feats)
                        results_dict['body_part'].append(body_part)
                        results_dict['interp'].append(interp)
                        results_dict['emotion'].append(emotion)
                        results_dict['fold'].append(fold)
                        results_dict['epoch'].append(best_epoch)

                scores_per_fold = []
                for i, all_scores in enumerate(all_scores_this_experiment):
                    for fold in all_scores.keys():
                        fold_scores = all_scores[fold][1][best_epoch]
                        fold_acc = all_scores[fold]['acc'][best_epoch]
                        scores_per_fold.append(np.concatenate([fold_scores, fold_acc], axis=1))
                        results_dict['p'].append(fold_scores[0])
                        results_dict['p_std'].append(None)
                        results_dict['r'].append(fold_scores[1])
                        results_dict['r_std'].append(None)
                        results_dict['f'].append(fold_scores[2])
                        results_dict['f_std'].append(scores[2])
                        result_dict['acc'].append(acc)
                        result_dict['acc_std'].append(None)
                    # now handle the average
                    av_scores = av_scores_this_experiment[i][best_epoch]
                    stds = np.std(np.array(scores_per_fold, axis=1))
                    results_dict['model'].append(model)
                    results_dict['feats'].append(feats)
                    results_dict['body_part'].append(body_part)
                    results_dict['interp'].append(interp)
                    results_dict['emotion'].append(emotion)
                    results_dict['fold'].append('average')
                    results_dict['epoch'].append(best_epoch)
                    results_dict['p'].append(av_scores[0])
                    results_dict['p_std'].append(stds[0])
                    results_dict['r'].append(av_scores[1])
                    results_dict['r_std'].append(stds[1])
                    results_dict['f'].append(av_scores[2])
                    results_dict['f_std'].append(stds[2])
                    result_dict['acc'].append(av_scores[3])
                    result_dict['acc_std'].append(stds[3])

pd.to_csv('all_results.csv', results_dict)
