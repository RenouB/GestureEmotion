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
df = df.groupby(['region', 'stat'], as_index=False).mean()[["stat","region","anger","happiness","sadness","surprise"]]

fig, ax = plt.subplots(2,2, figsize=(16,18))
emotions = ["anger","happiness","sadness","surprise"]
for emotion in emotions:
    cat = sns.catplot(x="region", y=emotion, hue="stat", data=df, kind="bar",
            hue_order=["mean","max","min","rel_max","displacement","variance","missing"],
            order=["full_hh","head","hands"], palette=sns.color_palette("hls", 8))
    cat.set(xlabel="body part region", ylabel="correlation",ylim=(-10,10))
    plt.title(emotion,loc='right')

plt.show()
