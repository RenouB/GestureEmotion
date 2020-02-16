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
from models.emotion_classification.data.datasets import PoseDataset
from argparse import ArgumentParser
SCORES_DIR = os.path.join(PROJECT_DIR, "models/emotion_classification")
print(SCORES_DIR)
emotions=[(0,"anger"),(1,"happiness"),(2,"sadness"),(3,"surprise")]
body_parts=["full","full-hh","head","hands"]

att_weights_dct = {"datapoint":[],"model":[], "actor":[],"body_part":[],
                    "anger0":[], "happiness0":[], "sadness0":[],"surprise0":[],
                    "anger1":[], "happiness1":[], "sadness1":[],"surprise1":[]}

predictions_dct = {"datapoint":[],"model":[], "actor":[], "body_part":[], "anger":[],"happiness":[],
                    "sadness":[], "surprise":[]}


def get_input_dim(keypoints, input):
	if keypoints == 'full':
		dim = len(constants["WAIST_UP_BODY_PART_INDICES"]) * 2
	if keypoints == 'full-hh':
		dim = len(constants["FULL-HH"]) * 2
	if keypoints == 'full-head':
		dim = len(constants["FULL-HEAD"]) * 2
	if keypoints == 'head':
		dim = len(constants["HEAD"]) * 2
	if keypoints == 'hands':
		dim = len(constants["HANDS"]) * 2

	if input == 'deltas-noatt':
		return dim * 3

	return dim

if __name__ == "__main__":
    for model in ["BiLSTM", "JointBiLSTM"]:

        data_length = 14839*2
        for body_part in body_parts:
            if model == "JointBiLSTM":
                att_weights_dct["body_part"]+= [body_part]*data_length
                att_weights_dct["model"]+= [body_part]*data_length
                att_weights_dct["datapoint"] += [i for i in range(data_length)]
                att_weights_dct["actor"] += ['A']*(data_length // 2) + ['B']*(data_length // 2)
            predictions_dct["body_part"]+= [body_part]*data_length
            predictions_dct["model"]+= [model]*data_length
            predictions_dct["datapoint"] += [i for i in range(data_length)]
            predictions_dct["actor"] += ['A']*(data_length // 2) + ['B']*(data_length // 2)

            for emotion_index, emotion_str in emotions:
                if emotion_index == 0:
                    fold = 'fold6.weights'
                elif emotion_index == 1:
                    fold = 'fold1.weights'
                elif emotion_index == 2:
                    fold = 'fold7.weights'
                elif emotion_index == 3:
                    fold = 'fold7.weights'

                print(model, body_part, emotion_str)
                # initialize dataset, model
                input_dim = get_input_dim(body_part, 'brute')
                if model == "BiLSTM":
                    data = PoseDataset(interval=3, seq_length=5, keypoints=body_part,
                                        joint = False, emotion=emotion_index, input='brute',
                                        interp=False)
                    net = BiLSTM(input_dim, hidden_dim=60,lstm_layer=2,dropout=0.5)
                    weights_name = 'interp-True-IND-0-'+body_part+ \
                                    '-lr0.001-l20.001-dr0.5-ep70'
                    ind_or_joint ='ind/pose'

                elif model == "JointBiLSTM":
                    data = PoseDataset(interval=3, seq_length=5, keypoints=body_part,
                                        joint=True, emotion=emotion_index, input='brute',
                                        interp=False)
                    net = JointBiLSTM(input_dim, hidden_dim=60,attention_dim=60,
                                lstm_layer=2,dropout=0.5)
                    weights_name = 'interp-True-JOINT-0-'+body_part+ \
                                    '-lr0.001-l20.001-dr0.5-ep70'
                    ind_or_joint = 'joint/pose/'

                weights_dir = os.path.join(SCORES_DIR, model, 'brute', emotion_str,
                                            ind_or_joint, 'weights/')

                print(len(data))
                print(weights_name)
                print(fold)
                file = [file.path for file in os.scandir(weights_dir) if file.name.startswith(weights_name)
                        and file.name.endswith(fold)][0]
                try:
                    net.load_state_dict(torch.load(file, map_location=lambda storage, loc: storage))
                except FileNotFoundError:
                    print("weights not found")
                    print(file)
                    continue
                net.double()
                net.eval()
                print("Model loaded")
                # data = Subset(data, [i for i in range(200)])
                loader = DataLoader(data, batch_size=len(data))
                for batch in loader:
                    if model == "BiLSTM":
                        out, _ = net(batch['pose'].double())
                        predictions = (out >= 0.5).int()[:,0].tolist()
                        predsA = predictions[:int(len(predictions) / 2)]
                        predsB = predictions[int(len(predictions) / 2):]
                        print("PREDS", len(predictions))
                    else:
                        outA, outB, attA, attB = net(batch['poseA'].double(), batch['poseB'].double())
                        # print(attA.shape, attB.shape)
                        predsA = (outA >= 0.5).int()[:,0].tolist()
                        predsB = (outA >= 0.5).int()[:,0].tolist()
                        # print(attA)
                        # print(attB)
                        attA = attA.reshape(-1,2).numpy()
                        attB = attB.reshape(-1,2).numpy()
                        print(attA.shape, attB.shape)
                        print(len(attA[:,1].tolist() + attB[:,0].tolist()))
                        att_weights_dct[emotion_str+"0"] += attA[:,0].tolist() \
                                                        +attB[:,0].tolist()
                        att_weights_dct[emotion_str+"1"] += attA[:,1].tolist() \
                                                        +attB[:,1].tolist()
                    predictions_dct[emotion_str] += predsA+predsB

                    print('inference complete')

for key, item in att_weights_dct.items():
    print(key,len(item))
for key, item in predictions_dct.items():
    print(key,len(item))


att_df = pd.DataFrame(att_weights_dct)
att_df.to_csv("attention_weights.csv")
preds_df = pd.DataFrame(predictions_dct)
preds_df.to_csv("predictions.csv")
