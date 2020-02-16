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


region_indices = {"full_hh" : constants["FULL-HH"],
                    "head" : constants["HEAD"],
                    "hands" : constants["HANDS"]}

df = pd.read_csv("body_stat_correlations.csv")
df = df[(df.stat != "kurtosis") & (df.stat != "skewness")]
print(len(df))
body_parts = df.body_part.tolist()
regions = []
for part in body_parts:
    appended=False
    for region, indices in region_indices.items():
        if part in indices:
            regions.append(region)
            appended = True
    if appended == False:
        print(region_indices)
        print('not appended', part)
print(len(df.body_part))
print(len(regions))
df["region"] = regions
df = df.groupby(['region', 'stat'], as_index=False).mean()[["stat","region","anger","happiness","sadness","surprise","p_anger","p_happiness","p_sadness","p_surprise"]]
df = df.reset_index()

fig, ax = plt.subplots(1,1, figsize=(16,18))
emotions = ["anger","happiness","sadness","surprise"]


for region in df.region.unique():
    blah = df[df.region == region]
    print(region)
    print('diff anger', (abs(blah.anger - blah.p_anger)).mean())
    print('diff happiness', (abs(blah.happiness - blah.p_happiness)).mean())
    print('diff sadness', (abs(blah.sadness - blah.p_sadness)).mean())
    print('diff surprise', (abs(blah.surprise - blah.p_surprise)).mean())
print('mean')
print('diff anger', (abs(df.anger - df.p_anger)).mean())
print('diff happiness', (abs(df.happiness - df.p_happiness)).mean())
print('diff sadness', (abs(df.sadness - df.p_sadness)).mean())
print('diff surprise', (abs(df.surprise - df.p_surprise)).mean())

#
# for emotion in emotions:
#
#
#     cat = sns.catplot(x="region", y='p_'+emotion, hue="stat", data=df, kind="bar",
#             hue_order=["mean","max","min","rel_max","displacement","variance","missing"],
#             order=["full_hh","head","hands"], palette=sns.color_palette("hls", 7))
#     cat.set(xlabel="body part region", ylabel="correlation",ylim=(-10,10))
#     plt.subplots_adjust(top=0.9)
#     plt.suptitle(emotion, fontsize = 16)
#
#
#     ax = plt.gca()
#     y_coords = []
#     x_mins = []
#     x_maxes = []
#     for region in ['full_hh', 'head', 'hands']:
#         for stat in ['mean','max','min','rel_max','displacement','variance','missing']:
#             print("##################")
#             print(region, stat, emotion)
#             print(df[(df.region == region) & (df.stat == stat)]['p_'+emotion])
#             y_coords.append(df[(df.region == region) & (df.stat == stat)][emotion])
#     # print(len(y_coords))
#     # print(y_coords)
#     x_coords = []
#     colors = ["#7d120a", "#785f04", "#047516", "#04756d", "#043575", "#4f0475", "#750462"] * 3
#     print(len(colors))
#     patches = sorted(ax.patches, key=lambda e: e.get_x())
#     for i, p in enumerate(patches):
#         ax.hlines(y=y_coords[i], xmin=p.get_x(), xmax=p.get_x()+p.get_width(), linewidth=3, colors=colors[i])
#
#     # ax.hlines(y_coords, x_mins, x_maxes)
#
#
# plt.show()
#
# # emotions = ["p_anger","p_happiness","p_sadness","p_surprise"]
# # for emotion in emotions:
# #     cat = sns.catplot(x="region", y=emotion, hue="stat", data=df, kind="bar",
# #             hue_order=["mean","max","min","rel_max","displacement","variance","missing"],
# #             order=["full_hh","head","hands"], palette=sns.color_palette("hls", 8))
# #     cat.set(xlabel="body part region", ylabel="correlation",ylim=(-10,10))
# #     plt.subplots_adjust(top=0.9)
# #     plt.suptitle(emotion, fontsize = 16)
# #
# # #
