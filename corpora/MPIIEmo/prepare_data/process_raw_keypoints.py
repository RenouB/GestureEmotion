import json
import numpy as np
import os
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]
# function to centre list of keypoints, transform into tuple

# function to get MSE between two sets of keypoints; used to identify to which person
# from time t - 1 a set of keypoints at time t belongs

for view_folder in os.listdir(RAW_BODY_FEATS_DIR):
    video_annotations = {'A' : [], 'B' : []}
    #TODO: Find way of determining who is actor A, actor B
    for video in oslistdir(os.path.join(RAW_BODY_FEATS_DIR, view_folder):)
        for video_json os.listdir(os.path.join(RAW_BODY_FEATS_DIR, view_folder, video_json)):
            with open(os.path.join(RAW_BODY_FEATS_DIR, view_folder, video_json)) as f:
                js = json.load(f)
            people = js['people']
            if len(people) == 2:
                one, two = people
            else:
                mseA =
