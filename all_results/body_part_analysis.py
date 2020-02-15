import pickle
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]
sys.path.insert(0, MODELS_DIR)

df = pd.read_csv("all_results.csv")
df = df[(df.shuffle == False) & (df.fold == "average") & (df.body_part.isnull() == False)]
only_model_settings = df[["model","feats","interp","emotion"]]
unique_param_combinations = list(only_model_settings.values)

emotions = df.emotion.unique()
models = ["CNN","BiLSTM","JointBiLSTM"]
index_to_body_part = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:15, 9:16, 10:17, 11:18}

body_parts = df.body_part.unique()
# cool! so now I have  list containing all unique model hyperparam sets
# param set: 0; model, 1; feats, 2;interp, 3;emotion

emotion_wise_part_analysis_dicts = {emotion:{model:{body_part:[] for body_part in body_parts}
                                    for model in models}
                                    for emotion in emotions}
# aggregate_body_part_analysis_dict = {}
for model in models:
    for emotion in emotions:
        this_emotion_dict = emotion_wise_part_analysis_dicts[emotion]
        model_param_sets = []
        for param_set in unique_param_combinations:
            if param_set[0] == model and param_set[-1] == emotion and \
                param_set.tolist() not in model_param_sets:
                model_param_sets.append(param_set.tolist())

        for param_set in model_param_sets:
            _, feats, interp, _ = param_set
            only_this_model = df[(df.model == model) & (df.feats == feats)
                                & (df.interp == interp) & (df.emotion == emotion)]
            only_this_model = only_this_model.sort_values(by='f',ascending=False).reset_index()

            for index, row in only_this_model.iterrows():
                part = row.body_part
                this_emotion_dict[model][part].append((row.f, index))

aggregate_body_part_analysis_dict = {emotion:{body_part:{'f':[],'f_std':[],
                                                        'rank':[], 'rank_std':[]}
                                                        for body_part in body_parts}
                                    for emotion in emotions}

for emotion, models in emotion_wise_part_analysis_dicts.items():
    for model, body_parts in models.items():
        for body_part, f_and_rank in body_parts.items():
            for item in f_and_rank:
                aggregate_body_part_analysis_dict[emotion][body_part]['f'].append(item[0])
                aggregate_body_part_analysis_dict[emotion][body_part]['rank'].append(item[1])
for emotion, body_parts in aggregate_body_part_analysis_dict.items():
        for body_part in body_parts:
            body_parts[body_part]['f_std'] = np.std(
                body_parts[body_part]['f'])
            body_parts[body_part]['f'] = np.mean(
                body_parts[body_part]['f'])
            body_parts[body_part]['rank_std'] = np.std(
                body_parts[body_part]['f'])
            body_parts[body_part]['rank'] = np.mean(
                body_parts[body_part]['rank'])

# final_agg_dict = {"emotion":[], "full-f":[], "full-rank":[], "full-head-f":[],
#                         "full-head-rank":[], "full-hh-f":[], "full-hh-rank":[], "head-f":[],
#                         "head-rank":[], "hands-f":[], "hands-rank":[]}
# for emotion, body_parts in aggregate_body_part_analysis_dict.items():
#     final_agg_dict["emotion"].append(emotion)
#     for body_part, metrics in body_parts.items():
#
#         final_agg_dict[body_part+'-f'].append((round(metrics['f']*100,2), round(metrics['f_std']*100,2)))
#         final_agg_dict[body_part+'-rank'].append((round(metrics['rank'],3), round(metrics['rank_std']*100,8)))
#
# final_agg_df = pd.DataFrame(final_agg_dict).round(decimals=2)

final_agg_dict = {"emotion":[], "body_part":[], "f":[], "f_std":[], "rank":[]}

for emotion, body_parts in aggregate_body_part_analysis_dict.items():
    for body_part, metrics in body_parts.items():
        final_agg_dict["emotion"].append(emotion)
        final_agg_dict["body_part"].append(body_part)
        final_agg_dict["f"].append(metrics['f']*100)
        final_agg_dict['f_std'].append(metrics['f_std']*100)
        final_agg_dict["rank"].append(metrics['rank'])

final_agg_df = pd.DataFrame(final_agg_dict).round(decimals=2)
print(final_agg_df)
cat = sns.catplot(x="f", y="body_part", hue="emotion", data=final_agg_df, kind="bar",
            hue_order=["anger","happiness","sadness","surprise"],
            order=["full","full-head","full-hh","head","hands"],
            orient="h",
            palette = ["#f54242","#f5cb42","#426cf5","#5ebf4d"])
cat.set(xlabel="Average F-Score", ylabel="Body part subset")

ranks=[]
for emotion in ["anger","happiness","sadness","surprise"]:
    for body_part in ["full","full-head","full-hh","head","hands"]:
        ranks.append(final_agg_df[(final_agg_df.emotion == emotion) & (final_agg_df.body_part == body_part)]["rank"])


ax = plt.gca()
for i, p in enumerate(ax.patches):
    print(p)
    print("x", p.get_x())
    print("y", p.get_y())
    print("width", p.get_width())
    print("height", p.get_height())
    print("\n")
    ax.text(p.get_width() + p.get_height()*10, p.get_y() + p.get_height() , "%.2f" % ranks[i],
            fontsize=10, color='black', ha='center', va='bottom')
plt.show()
