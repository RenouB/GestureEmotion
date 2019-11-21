import scipy.io
import pickle
import numpy as np
import os
import sys

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

''' converts raw_annotations.mat into a dictionary of
    column label : numpy array with annotations

    also converts to two seperate numpy arrays of different dimensionalities
    one contains the annotations, the other the information about annotated video '''

RAW_ANNOS_PATH = os.path.join(constants["MPIIEMO_ANNOS_DIR"],'raw_annotations.mat')
INFO_NAMES = ["scenario", "subscenario", "actorA", "actorB", "ratedActor", "videoTime"]
WRITE_DIR = constants["MPIIEMO_DATA_DIR"]

# load mat file, get annotations
mat = scipy.io.loadmat(RAW_ANNOS_PATH)
annos_mat = mat['annos']

# will convert annotations to dictionary
anno_dict = {}

# get column names
labels = list(annos_mat[0].dtype.names)
for label in labels:
    anno_dict[label] = annos_mat[label].item()

# get column names that correspond to information about wcorresponds to info or anno
# maintain lists of anno, info names; will pickle later
anno_names_for_pickle = []
info_names_for_pickle = []

for name in anno_dict.keys():
    if name not in INFO_NAMES:
        anno_names_for_pickle.append(name)
    else:
        info_names_for_pickle.append(name)

video_ids = []
subscene_ids =[]
A_or_B = []

for i in range(len(anno_dict['scenario'])):
    scene, subscene_number, actorA, actorB = [str(anno_dict[key][i].item()) for key in ['scenario', 'subscenario', 'actorA', 'actorB']]
    scene = scene[2:-2]
    if str(anno_dict['ratedActor'][i]) == actorA:
        A_or_B.append('A')
    else:
        A_or_B.append('B')

    if len(actorA) == 1:
        actorA = '0' + actorA
    elif actorA[0] == '2':
        actorA = '0'+ actorA[1]

    if len(actorB) == 1:
        actorB = '0' + actorB
    elif actorB[0] == '2':
        actorB = '0'+ actorB[1]

    actorA = 'A' + actorA
    actorB = 'B' + actorB
    video_id = '_'.join([scene, subscene_number, actorA, actorB])
    subscene_id = '_'.join([scene, subscene_number])
    video_ids.append(video_id)
    subscene_ids.append(subscene_id)

anno_dict['video_ids'] = np.array(video_ids)
anno_dict['subscene_ids'] = np.array(subscene_ids)
anno_dict['A_or_B'] = np.array(A_or_B)

video_ids = []
subscene_ids = []

# verify that this has been done correctly by comparing contents
# of dict and numpy arrays
with open(os.path.join(WRITE_DIR, 'annotations_dict.pkl'), 'wb') as f:
    pickle.dump(anno_dict, f)
