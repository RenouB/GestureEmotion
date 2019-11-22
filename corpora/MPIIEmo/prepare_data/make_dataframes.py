import pickle
import numpy as np
import os
import sys
import pandas as pd

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

RAW_ANNOS_PATH = os.path.join(constants["MPIIEMO_ANNOS_DIR"],'raw_annotations.mat')
INFO_NAMES = ["scenario", "subscenario", "actorA", "actorB", "ratedActor", "videoTime"]
WRITE_DIR = constants["MPIIEMO_DATA_DIR"]

with open(os.path.join(WRITE_DIR, "annotations_dict.pkl"), 'rb') as f:
    annos = pickle.load(f)

all_dfs = []
for i in range(5):
    all_dfs.append({key:[] for key in annos.keys()})

for key, array in annos.items():
    for i in range(5):
        try:
            all_dfs[i][key] = array[:,i]
        except:
            try:
                all_dfs[i][key] = array.squeeze(axis=1)
            except:
                all_dfs[i][key] = array

all_dfs = [pd.DataFrame(df) for df in all_dfs]
for i, df in enumerate(all_dfs):
    df["A_or_B"] = np.where(df["ratedActor"] == df["actorA"], "A", "B")
    df.to_csv(os.path.join(WRITE_DIR, "df{}.csv".format(i)))
