import sys
import os
import pydrive
import numpy
import json
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
from process_utils import map_new_pose_to_person, centre_body_keypoints, map_person_to_a_or_b

'''
the download portion of this script doesnt yet work
the rest  will go through all views, all videos, extract body poses from openpose
json output. it will assign each each pose to either person a or b,
centre around 0 and scale
'''

# download section - not working
# gauth = GoogleAuth()
# # Try to load saved client credentials
# # gaut
# h.LoadCredentialsFile("mycreds.txt")
# # if gauth.credentials is None:
# #     # Authenticate if they're not there
# #     gauth.LocalWebserverAuth()
# # elif gauth.access_token_expired:
# #     # Refresh them if expired
# #     gauth.Refresh()
# # else:
# #     # Initialize the saved creds
# #     gauth.Authorize()
# # Save the current credentials to a file
# gauth.LocalWebserverAuth()
# drive = GoogleDrive(gauth)
# gauth.SaveCredentialsFile("mycreds.txt")
#
# RAW_BODY_FEATS_DRIVE_ID = constants["RAW_BODY_FEATS_DRIVE_ID"]
# download = drive.CreateFile({'id': RAW_BODY_FEATS_DRIVE_ID})
# print("Downloading")
# download.GetContentFile("body_feats/raw/all")

# get relevant directories
PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]

# initialize dictionary for to hold body keypoints from each camera view
# for all videos
all_videos = {}

# begin looping through view folders
for current_view in [item for item in os.listdir(RAW_BODY_FEATS_DIR) if item != "all"]:
    print("#############", current_view, "##################")
    # loop through video in folder
    for current_video in os.listdir(os.path.join(RAW_BODY_FEATS_DIR, current_view)):
        print("#############", current_video, "###############")
        person1_poses = {}
        person2_poses = {}
        # these keep track of detected poses from last frame - initialize them
        # arbitrarily big value
        last_person1 = [10000]*75
        last_person2 = [10000]*75
        # begin looping through json for this video
        for current_json in sorted(os.listdir(os.path.join(RAW_BODY_FEATS_DIR, current_view,
                current_video))):
            # print(current_json)
            frame_id = int(current_json.split('_')[4][-3:])
            # print("#############", frame_id, "#############")
            with open(os.path.join(RAW_BODY_FEATS_DIR, current_view, current_video,
                    current_json)) as f:
                # load json and extract poses
                js = json.load(f)
                people = js['people']
                # if there are no people in frame, put None
                if not len(people):
                    person1_poses[frame_id] = [10000]*75
                    person2_poses[frame_id] = [10000]*75

                # check if this is first time anyone has been detected
                if not last_person1:
                    # if only one person, they become person one
                    if len(people) == 1:
                        person1_poses[frame_id] = people[0]["pose_keypoints_2d"]
                        last_person1 = people[0]["pose_keypoints_2d"]
                        current_person1 = people[0]["pose_keypoints_2d"]
                        person2_poses[frame_id] = [10000]*75
                        last_person2 = [10000]*75
                        current_person2 = [10000]*75
                        assignment = (1, 1)
                    # if two people, assign persons one and two
                    if len(people) == 2:
                        person1_poses[frame_id] = people[0]["pose_keypoints_2d"]
                        current_person1 = people[0]["pose_keypoints_2d"]
                        last_person1 = people[1]["pose_keypoints_2d"]
                        person2_poses[frame_id] = people[1]["pose_keypoints_2d"]
                        current_person2 = people[1]["pose_keypoints_2d"]
                        last_person2 = people[1]["pose_keypoints_2d"]
                        assignment = (1, 1)
                # if there's only one person and this isn't the first person
                elif len(people) == 1:
                    current_person1 = people[0]["pose_keypoints_2d"]
                    current_person2 = [10000]*75
                    assignment = \
                        map_new_pose_to_person(last_person1, last_person2,
                        current_person1, current_person2)

                # if there are two people that aren't the first people
                elif len(people) == 2:
                    current_person1 = people[0]["pose_keypoints_2d"]
                    current_person2 = people[1]["pose_keypoints_2d"]
                    assignment = \
                        map_new_pose_to_person(last_person1, last_person2,
                        current_person1, current_person2)

                # update pose dictionaries according to assignment result
                if assignment == (1,1):
                    last_person1 = current_person1
                    person1_poses[frame_id] = np.array([current_person1[i:i+2] for i in range(0, 73, 3)])
                    last_person2 = current_person2
                    person2_poses[frame_id] = np.array([current_person2[i:i+2] for i in range(0, 73, 3)])

                elif assignment == (1,2):
                    last_person2 = current_person1
                    person2_poses[frame_id] = np.array([current_person1[i:i+2] for i in range(0, 73, 3)])
                    last_person1 = current_person2
                    person1_poses[frame_id] = np.array([current_person2[i:i+2] for i in range(0, 73, 3)])

        # finally, once we have sequence of all poses for persons 1 and 2, map
        # these back to actors A or B from annotations
        a_or_b = map_person_to_a_or_b(person1_poses, person2_poses)
