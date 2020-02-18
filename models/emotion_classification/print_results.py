import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
df = pd.read_csv("all_results.csv")
av_df = df[(df.fold == "average")]
folds_df = df[(df.fold != "average")]

foldwise_f_scores = {"rand":{}, "SVM":{}, "Linear":{}, "CNN":{}, "attCNN":{},
                "BiLSTM":{}, "JointBiLSTM":{}}
av_f_scores = {"rand":{}, "SVM":{}, "Linear":{}, "CNN":{}, "attCNN":{},
                "BiLSTM":{}, "JointBiLSTM":{}}


for model in ["rand", "SVM", "Linear", "CNN", "attCNN", "BiLSTM", "JointBiLSTM"]:
    for emotion in ["anger", "happiness", "sadness", "surprise"]:
        model_key = model
        if model == "attCNN":
            feats = "deltas"
            body_part = "full"
        elif model == "SVM" or model == "Linear":
            feats = "stats"
            body_part = np.NaN
        elif model == "rand":
            feats = "no-input"
            body_part = np.NaN
        else:
            feats = "brute"
            body_part = "full"

        av_model_results = av_df[(av_df.emotion == emotion) & (av_df.model == model_key)
                            & (av_df.feats == feats) & ((av_df.body_part == body_part)
                            | (av_df.body_part.isna()))]
        if len(av_model_results) == 0:
            continue

        foldwise_model_results = folds_df[(folds_df.emotion == emotion) & (folds_df.model == model_key)
                            & (folds_df.feats == feats) & ((folds_df.body_part == body_part)
                            | (folds_df.body_part.isna()))]


        av_f_scores[model][emotion] = (av_model_results.f.iloc[0], av_model_results.f_std.iloc[0])
        foldwise_f_scores[model][emotion] = list(foldwise_model_results.f)

print("GLOBAL RESULTS TABLE")
print("Reporting F1 score on the positive emotion class, averaged across all eight folds.")
print("Features used: attCNN; deltas. BiLSTM, JointBiLSTM, CNN: brute. SVM, Linear: statistical")
print("results reported for attCNN, BiLSTM, JointBILSTM and CNN are from FULL body part models.")
print("{:>14} {:>14} {:>14} {:>14} {:>14} {:>14}".format("model", "anger", "happiness", "sadness", "surprise", "mean"))
for model in ["rand", "SVM", "Linear", "CNN", "attCNN", "BiLSTM", "JointBiLSTM"]:
    to_print = "{:<14}".format(model)
    mean_f = 0
    mean_std = 0
    all_f_scores = []
    for emotion in ["anger", "happiness", "sadness", "surprise"]:
        if model not in av_f_scores or emotion not in av_f_scores[model]:
            continue
        # print(av_f_scores[model])
        to_print += "{:<6.2f}/ {:<6.2f}{:3}".format(av_f_scores[model][emotion][0]*100,
                                                av_f_scores[model][emotion][1]*100,"")
        mean_f += av_f_scores[model][emotion][0]
        mean_std = av_f_scores[model][emotion][1]
        all_f_scores += foldwise_f_scores[model][emotion]
    all_f_scores = np.array(all_f_scores)
    mean = all_f_scores.mean()
    std = all_f_scores.std()
    to_print += "{:<6.2f} / {:<6.2f}".format(mean*100, std*100)
    print(to_print)

#p values

print("\n\nP values between different model outputs, for all emotions and per emotion.")
models = ["rand", "SVM","Linear","CNN","attCNN","BiLSTM","JointBiLSTM"]
foldwise_fs_per_model = {}
for model in models:
    if model not in foldwise_fs_per_model:
        foldwise_fs_per_model[model] = []
        for emotion in ["anger", "happiness", "sadness", "surprise"]:
            if model not in foldwise_f_scores or emotion not in foldwise_f_scores[model]:
                continue
            foldwise_fs_per_model[model] += foldwise_f_scores[model][emotion]


model_p_df = {}
for model1 in models:
    if model1 not in model_p_df:
        model_p_df[model1] = []
    for model2 in models:
        model_p_df[model1].append(ttest_ind(foldwise_fs_per_model[model1],
                                    foldwise_fs_per_model[model2], equal_var=False)[1])
model_p_df = (pd.DataFrame(model_p_df, index=models)*100).round(decimals=2)
print("\nGLOBAL")
print(model_p_df)
emotion_p_dfs = {}
for emotion in ["anger", "happiness", "sadness", "surprise"]:
    emotion_p_dict = {"rand":[],"SVM":[],"Linear":[], "CNN":[], "attCNN":[],
                        "BiLSTM":[], "JointBiLSTM":[]}
    for model1 in ["rand", "SVM","Linear","CNN","attCNN","BiLSTM","JointBiLSTM"]:
        for model2 in ["rand", "SVM", "Linear", "CNN","attCNN","BiLSTM","JointBiLSTM"]:
            emotion_p_dict[model1].append(ttest_ind(foldwise_f_scores[model1][emotion],
                                                foldwise_f_scores[model2][emotion], equal_var = False)[1])
    emotion_p_dfs[emotion] = (pd.DataFrame(emotion_p_dict,index=models)*100).round(decimals=2)
    print("\n")
    print(emotion.upper())
    print(emotion_p_dfs[emotion])
