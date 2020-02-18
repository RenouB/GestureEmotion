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

predictions_df = pd.read_csv("predictions.csv")
predictions_df = predictions_df[(predictions_df.model == "JointBiLSTM") & (predictions_df.body_part == "full")]
predictions_df = predictions_df.loc[predictions_df.index.repeat(24)]



df = pd.read_csv("body_keypoint_stats.csv")
print(len(df))
print(len(predictions_df))
df["p_anger"] = predictions_df.anger.tolist()
df["p_happiness"] = predictions_df.happiness.tolist()
df["p_sadness"] = predictions_df.sadness.tolist()
df["p_surprise"] = predictions_df.surprise.tolist()

correlations_df = {"body_part":[], "stat":[], "anger":[],
                    "happiness":[], "sadness":[], "surprise":[], "p_anger":[],
                    "p_happiness":[], "p_sadness":[], "p_surprise":[]}
for body_part in df.body_part.unique():
    for stat in ["mean","min", "max","variance", "kurtosis","skewness", "rel_max",
     "displacement"]: # "missing",
        correlations = [0,0,0,0,0,0,0,0]
        for dim in ["x","y"]:
            subset = df[(df.body_part == body_part) & (df.dim == dim)]

            for i, emotion in enumerate(["anger", "happiness", "sadness", "surprise",
                                        "p_anger", "p_happiness", "p_sadness",
                                        "p_surprise"]):

                correlations[i] += (stats.pearsonr(subset[stat], subset[emotion])[0])

        correlations = np.array(correlations) / 2
        correlations_df["body_part"].append(body_part)
        correlations_df["stat"].append(stat)
        correlations_df["anger"].append(round(correlations[0]*100,2))
        correlations_df["happiness"].append(round(correlations[1]*100,2))
        correlations_df["sadness"].append(round(correlations[2]*100,2))
        correlations_df["surprise"].append(round(correlations[3]*100,2))
        correlations_df["p_anger"].append(round(correlations[4]*100,2))
        correlations_df["p_happiness"].append(round(correlations[5]*100,2))
        correlations_df["p_sadness"].append(round(correlations[6]*100,2))
        correlations_df["p_surprise"].append(round(correlations[7]*100,2))

df = pd.DataFrame(correlations_df)
df.to_csv("body_stat_correlations.csv")
print(df)
