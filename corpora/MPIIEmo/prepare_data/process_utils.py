import numpy as np
import os, sys
import matplotlib.pyplot as plt
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

def plot_body_keypoints(body_keypoints_as_tuples):
    connection_mapping = constants["PAFS"]
    for start, end in connection_mapping:
        start_x, start_y, _ = body_keypoints_as_tuples[start]
        end_x, end_y, _ = body_keypoints_as_tuples[end]
        plt.plot([start_x, end_x], [start_y, end_y])
    plt.show()

def convert_body_keypoints_to_tuples(body_keypoints):
    as_tuples =[]
    print(len(body_keypoints))
    working = []
    for i, keypoint in enumerate(body_keypoints):
        if i % 3 == 0:
            print('assigning x')
            working.append(keypoint)
        if i % 3 == 2:
            print('assigning y')
            working.append(-1*keypoint)
        if i % 3 == 1:
            print('assigning confidence')
            working.append(-keypoint)
        if len(working) == 3:
            as_tuples.append(tuple(working))
            working =[]
    print(as_tuples)
    return as_tuples
# adjust all body keypoint coordinates
def centre_body_keypoints(body_keypoints):
    center_x, center_y = body_keypoints[24:26]
    centered_keypoints = []
    for i, keypoint in enumerate(body_keypoints):
        if i % 3 == 0:
            centered_keypoints.append(keypoint - center_x)
        elif i % 3 == 1:
            centered_keypoints.append(keypoint - center_y)
        elif i % 3 == 2:
             centered_keypoints.append(keypoint)

    return centered_keypoints
#
# def mse_between_keypoints(body_keypoints1, body_keypoints2):
#     keypoint_arrays = []
#     for keypoints in [body_keypoints1, body_keypoints2]:
#         keypoints_as_list = []
#         for i, keypoint in enumerate(keypoints):
#             if i % 3 == 0:
#                 x = keypoint
#             if i % 3 == 1:
#                 y = keypoint
#             if i % 3 == 2:
#                 keypoints_as_list.append([x,y])
#             keypoint_arrays.append(np.array(keypoints_as_list))
#
#     return (keypoint_arrays[0] - keypoint_arrays[1])**2.mean(axis=0)
