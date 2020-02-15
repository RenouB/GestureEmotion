import pickle
import numpy as np
import pandas as pd
import os, sys
import torch
from torch.utils.data import DataLoader, Subset
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-2])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)
from models.emotion_classification.BiLSTM.bilstm import BiLSTM
from models.emotion_classification.JointBiLSTM.joint_bilstm import JointBiLSTM
from models.emotion_classification.data.torch_datasets import PoseDataset
from argparse import ArgumentParser
from sklearn.metrics import precision_recall_fscore_support
SCORES_DIR = os.path.join(PROJECT_DIR, "all_results/score_files")
print(SCORES_DIR)
emotions=["anger", "happiness", "sadness", "surprise"]
body_parts=["full","full-hh","head","hands"]


# first look at predictions
labels_df = pd.read_csv()
df = pd.read_csv("predictions.csv")
# print(df.head())
for model in ["BiLSTM","JointBiLSTM"]:
    for body_part in body_parts:
        for actor in ['A','B']:
            subset = df[(df.model == model) & (df.body_part == body_part) & (df.actor == actor)][["anger","happiness","sadness","surprise"]]
            print(model, body_part, actor)

            for emotion in emotions:
                print(emotion, subset[emotion].unique())
            print("\n")

df = pd.read_csv("attention_weights.csv")
for body_part in body_parts:
    for actor in ["A","B"]:
        subset = df[(df.body_part == body_part) & (df.actor == actor)]
        print(model, body_part, actor)
        for emotion in emotions:
            print(emotion, len(subset[emotion+'0'].unique()), subset[emotion+'0'].mean(),
                    len(subset[emotion+'1'].unique()), subset[emotion+"1"].mean())
        print("\n")
