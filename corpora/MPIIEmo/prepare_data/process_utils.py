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


def plot_body_keypoints(body_keypoints_as_tuples):
    '''not yet functional. for visualization of body keypoints'''
    connection_mapping = constants["PAFS"]
    for start, end in connection_mapping:
        start_x, start_y, _ = body_keypoints_as_tuples[start]
        end_x, end_y, _ = body_keypoints_as_tuples[end]
        plt.plot([start_x, end_x], [start_y, end_y])
    plt.show()

def convert_body_keypoints_to_tuples(body_keypoints):
    '''deprecated. converts body keypoints to tuples for easier indexing'''
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
    '''not yet completed. centres body keypoints by setting mid hip to 0'''
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

def map_new_pose_to_person(last_person1, last_person2, current_person1, current_person2):
    '''openpose doesn't track to which person a skeleton belongs
       we do this manually by assigning a current person's skeleton to the last
       skeleton that is closest
       param: last_person1, last_person2: body keypoints from last frame
       param: current_person1, current_person2: body keypoints from current frame
       return: tuple mapping current_person1 to correct last person (1 or 2)
     '''

    # convert keypoints to arrays
    current1_keypoints = [np.array(current_person1[i:i+2]) for i in range(0, 73, 3)]
    # checks if current_person2 exists
    if sum(current_person2) == 75*10000:
        current2_keypoints = [np.array(current_person2[i:i+2]) for i in range(0, 73, 3)]
    else:
        current2_keypoints = [np.array([10000, 10000]) for i in range(25)]

    last1_keypoints = [np.array(last_person1[i:i+2]) for i in range(0, 73, 3)]

    # checks if last_person2 exists
    if sum(last_person2) == 75*10000:
        last2_keypoints = [np.array(last_person2[i:i+2]) for i in range(0, 73, 3)]
    else:
        last2_keypoints = [np.array([10000,10000]) for i in range(25)]

    # begin trying cosine similarity
    # current1_ar = np.array(current1_keypoints).flatten()
    # last1_ar = np.array(last1_keypoints).flatten()
    # last2_ar = np.array(last2_keypoints).flatten()
    # cosine11 = cosine(current1_ar, last1_ar)
    # cosine12 = cosine(current1_ar, last2_ar)
    #
    # if cosine11 > cosine12:
    #     return (1,1)
    # else:
    #     return(1,2)

    # for comparison, find all body parts which are detected in all skeletons
    common_indices = [i for i in range(len(current1_keypoints)) if
                        current1_keypoints[i].sum() != 20000 and current2_keypoints[i].sum() != 20000
                        and last1_keypoints[i].sum() != 20000  and last2_keypoints[i].sum() != 20000
                        and i in STABLE_BODY_PART_INDICES]

    current1_common_keypoints = np.array([current1_keypoints[i] for i in common_indices])
    current2_common_keypoints = np.array([current2_keypoints[i] for i in common_indices])
    last1_common_keypoints = np.array([last1_keypoints[i] for i in common_indices])
    last2_common_keypoints = np.array([last2_keypoints[i] for i in common_indices])

    # fine distances from current_person1 to last_person1, current_person1 to
    #last_person2
    distance_c1_l1 = np.linalg.norm(current1_common_keypoints - \
                        last1_common_keypoints, axis=1).mean()
    distance_c1_l2 = np.linalg.norm(current1_common_keypoints - \
                        last2_common_keypoints, axis=1).mean()

    # if c1 closer to l1, assign 1 -> 1. else assign 1 -> 2
    if distance_c1_l1 > distance_c1_l2:
        return (1,2)
    else:
        return (1,1)

def map_person_to_a_or_b(person1_poses, person2_poses):
    '''Not yet completed. Maps set of poses from persons 1, 2 to labels A, B
    from annotations.
    param: personX_poses: dictionary with containing frame_id : pose from all
    frames
    '''
    first_fifteen_person1 = np.array([person1_poses[i] for i in range(30)])
    first_fifteen_person2 = np.array([person2_poses[i] for i in range(30)])
    print(first_fifteen_person1[:][8])
    print("###############")
    print(first_fifteen_person2[:][8])
    # get body parts that were detected in all frames for each skeleton
    print(first_fifteen_person1.shape)
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

    print("P1", total_displacement_person1)
    print("P2", total_displacement_person2)
    if total_displacement_person1 > total_displacement_person2:
        return {"A":1, "B":2}
    else:
        return {"A":2, "B":1}
