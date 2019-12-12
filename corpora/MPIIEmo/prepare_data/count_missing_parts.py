import sys
import os
import pydrive
import numpy
import json
import pickle
import numpy as np

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
from process_utils import map_new_pose_to_person, map_person_to_a_or_b, scale_keypoints, translate_keypoints

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
BODY_KEYPOINT_MAPPING = constants["BODY_KEYPOINT_MAPPING"]
with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'all_raw.pkl'), 'rb') as f:
    all_videos = pickle.load(f)

total_frames = 0
missing_counts = {i:0 for i in range(25)}
for video, views in all_videos.items():
    for view, actors in views.items():
        for actor, frames in actors.items():
            for keypoints in frames.values():
                total_frames += 1
                for i in range(len(keypoints)):

                    if keypoints[i].sum() == 0:
                        missing_counts[i] += 1
print("total posture frames:", total_frames)
for i, count in missing_counts.items():
    print(BODY_KEYPOINT_MAPPING[i], count)
