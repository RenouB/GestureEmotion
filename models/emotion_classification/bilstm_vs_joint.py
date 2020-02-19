import pickle
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
import scipy.stats as stats

preds_df = pd.read_csv("predictions.csv")

bilstm = preds_df[(preds_df.model == "BiLSTM") & (preds_df.body_part == 'full')]
joint = preds_df[(preds_df.model == "JointBiLSTM") & (preds_df.body_part == 'full')]

labels = pd.read_csv("body_keypoint_stats.csv")

labels = labels[(labels.body_part == 0) & (labels.dim == 'x')]

a_labels = labels[:len(labels) // 2]
b_labels = labels[len(labels) // 2:]
a_bilstm = bilstm[:len(labels) // 2]
b_bilstm = bilstm[len(labels) // 2:]
a_joint = joint[:len(labels) // 2]
b_joint = joint[len(labels) // 2:]
a_joint.reset_index(drop=True, inplace=True)
b_joint.reset_index(drop=True, inplace=True)
a_bilstm.reset_index(drop=True, inplace=True)
b_bilstm.reset_index(drop=True, inplace=True)
a_labels.reset_index(drop=True, inplace=True)
b_labels.reset_index(drop=True, inplace=True)


print(len(b_joint))
for df in [a_labels, b_labels, a_bilstm, b_bilstm, a_joint, b_joint]:
    df = df.reset_index()

emotions = ['anger','happiness','sadness','surprise']
things = ["joint", "bilstm", "labels"]
both_one = {emotion: {"joint":{},"bilstm":{}, "labels":{}} for emotion in emotions}
both_zero = {emotion: {"joint":{},"bilstm":{}, "labels":{}} for emotion in emotions}
different_10 = {emotion: {"joint":{},"bilstm":{}, "labels":{}} for emotion in emotions}
different_01 = {emotion: {"joint":{},"bilstm":{}, "labels":{}} for emotion in emotions}


a_as_lists = {}
b_as_lists = {}
for emotion in emotions:
    a_as_lists[emotion] = {'joint':a_joint[emotion].tolist(), 'bilstm':a_bilstm[emotion].tolist(),
                        'labels': a_labels[emotion].tolist()}
    b_as_lists[emotion] = {'joint':b_joint[emotion].tolist(), 'bilstm':b_bilstm[emotion].tolist(),
                    'labels': b_labels[emotion].tolist()}

print(emotions)
for emotion in emotions:
    both_one[emotion]['joint'] = [i for i in range(len(labels) // 2) if
                                    a_as_lists[emotion]['joint'][i] == 1 and
                                    b_as_lists[emotion]['joint'][i] == 1]
    both_one[emotion]['bilstm'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['bilstm'][i] == 1 and
                                b_as_lists[emotion]['bilstm'][i] == 1]
    both_one[emotion]['labels'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['labels'][i] == 1 and
                                b_as_lists[emotion]['labels'][i] == 1]
    both_zero[emotion]['joint'] = [i for i in range(len(labels) // 2) if
                                    a_as_lists[emotion]['joint'][i] == 0 and
                                    b_as_lists[emotion]['joint'][i] == 0]
    both_zero[emotion]['bilstm'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['bilstm'][i] == 0 and
                                b_as_lists[emotion]['bilstm'][i] == 0]
    both_zero[emotion]['labels'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['labels'][i] == 0 and
                                b_as_lists[emotion]['labels'][i] == 0]
    different_10[emotion]['joint'] = [i for i in range(len(labels) // 2) if
                                    a_as_lists[emotion]['joint'][i] == 1 and
                                    b_as_lists[emotion]['joint'][i] == 0]
    different_10[emotion]['bilstm'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['bilstm'][i] == 1 and
                                b_as_lists[emotion]['bilstm'][i] == 0]
    different_10[emotion]['labels'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['labels'][i] == 1 and
                                b_as_lists[emotion]['labels'][i] == 0]
    different_10[emotion]['joint'] = [i for i in range(len(labels) // 2) if
                                    a_as_lists[emotion]['joint'][i] == 0 and
                                    b_as_lists[emotion]['joint'][i] == 1]
    different_01[emotion]['bilstm'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['bilstm'][i] == 0 and
                                b_as_lists[emotion]['bilstm'][i] == 1]
    different_01[emotion]['labels'] = [i for i in range(len(labels) // 2) if
                                a_as_lists[emotion]['labels'][i] == 0 and
                                b_as_lists[emotion]['labels'][i] == 1]

# print(both_one)
emotions = ['anger','happiness','sadness','surprise']
for emotion in emotions:
    print("\n")
    print(emotion)
    for model in ["bilstm", "joint"]:
        print('\n')
        print(model)
        print("both one correct")
        try:
            print(len(set(both_one[emotion][model]).intersection(both_one[emotion]['labels'])) /len(both_one[emotion]['labels']))
        except: print("-------")
        print("both zero correct")
        try:
            print(len(set(both_zero[emotion][model]).intersection(both_zero[emotion]['labels'])) / len(both_zero[emotion]['labels']))
        except: print("-------")
        print("dif correct")
        try:
            print((len(set(different_10[emotion][model]).intersection(different_10[emotion]['labels'])) + len(set(different_01[emotion][model]).intersection(different_01[emotion]['labels']))) \
                / (len(different_10[emotion]['labels']) + len(different_01[emotion]['labels'])))
        except: print("-------")
