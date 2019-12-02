import numpy as np
import os, sys
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
# quick fix to get script to run from anywhere
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

''' contains many utilities used to process raw body features '''

# indices of stable body parts such as shoulder, head, hip, etc.
STABLE_BODY_PART_INDICES = constants["STABLE_BODY_PART_INDICES"]
WAIST_UP_BODY_PART_INDICES = constants["WAIST_UP_BODY_PART_INDICES"]

def translate_keypoints(body_keypoints):
    mid_hip = body_keypoints[8]
    translated_keypoints = []
    for keypoint in body_keypoints:
        if keypoint.sum() == 0:
            translated_keypoints.append(keypoint)
        else:
            translated_keypoints.append(keypoint - mid_hip)
    return translated_keypoints

def interpolate_missing_keypoints(body_keypoints, body_keypoints_sequence):
    # if half of the waist-up body parts are missing, eliminate this pose
    missing_keypoints = np.where(body_keypoints.sum(axis=1))
    if len(np.intersect1d(missing_keypoints, WAIST_UP_BODY_PART_INDICES)) > 7:
        return np.array([[10000,10000]*25])
    else:
        # start interpolating missing keypoints
        for i in range(len(body_keypoints_sequence)):


def scale_keypoints(body_keypoints):
    neck = body_keypoints[1]
    mid_hip = body_keypoints[8]
    l2 = np.linalg.norm(neck - mid_hip)
    print('################')
    print("L2:", l2)

    if l2 == 0:
        return body_keypoints
    return body_keypoints / l2

def map_new_pose_to_person(last_person1, last_person2, current_person1, current_person2):
    '''openpose doesn't track to which person a skeleton belongs
       we do this manually by assigning a current person's skeleton to the last
       skeleton that is closest
       param: last_person1, last_person2: body keypoints from last frame
       param: current_person1, current_person2: body keypoints from current frame
       return: tuple mapping current_person1 to correct last person (1 or 2)
     '''

    # convert keypoints to arrays
    current1_keypoints = np.array([np.array(current_person1[i:i+2]) for i in range(0, 73, 3)])
    current2_keypoints = np.array([np.array(current_person2[i:i+2]) for i in range(0, 73, 3)])
    last1_keypoints = np.array([np.array(last_person1[i:i+2]) for i in range(0, 73, 3)])
    last2_keypoints = np.array([np.array(last_person2[i:i+2]) for i in range(0, 73, 3)])

    common11 = [i for i in range(len(current1_keypoints)) if current1_keypoints[i].sum()
                    and last1_keypoints[i].sum() and i in STABLE_BODY_PART_INDICES]
    common12 = [i for i in range(len(current1_keypoints)) if current1_keypoints[i].sum()
                    and last2_keypoints[i].sum() and i in STABLE_BODY_PART_INDICES]

    distance11 = np.linalg.norm(current1_keypoints[common11].flatten() - last1_keypoints[common11].flatten()) / len(common11)
    distance12 = np.linalg.norm(current1_keypoints[common12].flatten() - last2_keypoints[common12].flatten()) / len(common12)

    if distance11 > distance12:
        assignment = (1,2)
    else:
        assignment = (1,1)

    return assignment

def map_person_to_a_or_b(person1_poses, person2_poses):
    '''Not yet completed. Maps set of poses from persons 1, 2 to labels A, B
    from annotations.
    param: personX_poses: dictionary with containing frame_id : pose from all
    frames
    '''
    first_fifteen_person1 = np.array([person1_poses[i] for i in range(30)])
    first_fifteen_person2 = np.array([person2_poses[i] for i in range(30)])
    # get body parts that were detected in all frames for each skeleton
    p1_detected_part_indices = np.where(first_fifteen_person1.min(0) > 0)
    p2_detected_part_indices = np.where(first_fifteen_person2.min(0) > 0)

    common_indices = np.intersect1d(p1_detected_part_indices, p2_detected_part_indices)
    common_indices = np.intersect1d(common_indices, STABLE_BODY_PART_INDICES)

    total_displacement_person1 = 0
    total_displacement_person2 = 0
    for row in range(1,30):
        for col in common_indices:
            total_displacement_person1 += abs(first_fifteen_person1[row - 1][col] -
                            first_fifteen_person1[row][col]).sum()
            total_displacement_person2 += abs(first_fifteen_person2[row - 1][col] -
                            first_fifteen_person2[row][col]).sum()

    if total_displacement_person1 > total_displacement_person2:
        return {"A":1, "B":2}
    else:
        return {"A":2, "B":1}
