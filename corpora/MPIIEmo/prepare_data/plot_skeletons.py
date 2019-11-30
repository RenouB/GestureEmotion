import pickle
import numpy as np
import os
import sys
import pandas as pd
import json
import pickle
from argparse import ArgumentParser
import matplotlib.pyplot as plt
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'all.pkl'),'rb') as f:
    all_videos = pickle.load(f)

def plot_body_keypoints(body_keypoints, color, alpha=1):
    connection_mapping = constants["PAFS"]
    for start, end in connection_mapping:
        start_x, start_y = body_keypoints[start]
        end_x, end_y = body_keypoints[end]
        if sum([start_x, start_y]) == 0 or sum([end_x, end_y]) == 0:
            continue
        plt.plot([start_x, end_x], [-start_y, -end_y], color=color, alpha=alpha)

def plot_frame(video, view, frame, actors, alpha=1):
    personA_keypoints = all_videos[video][view]['A'][frame]
    personB_keypoints = all_videos[video][view]['B'][frame]

    if actors == 3 or actors == 1:
        plot_body_keypoints(personA_keypoints, 'blue', alpha)
    if actors == 3 or actors == 2:
        plot_body_keypoints(personB_keypoints, 'red', alpha)

def plot_video(video, view, actors, framestep=15):
    personA_keypoints = all_videos[video][view]['A']
    total_frames = max(personA_keypoints.keys())
    for frame in range(0, total_frames, framestep):
        plot_frame(video, view, frame, actors, alpha=max(.1, frame / total_frames))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-video", action="store_true", default=True)
    parser.add_argument("-file", type=str)
    parser.add_argument("-view", type=str, default="view1")
    parser.add_argument("-actors", type=int, default=3)
    parser.add_argument("-framestep", type=int, default=15)
    parser.add_argument("-frame", type=int, default=1)
    args = parser.parse_args()

    if args.video:
        plot_video(args.file, args.view, args.actors, args.framestep)

    else:
        plot_frame(args.video_folder, args.view, args.frame, args.actors)

    plt.xlim(0, 1624)
    plt.ylim(-1224,0)
    plt.show()
