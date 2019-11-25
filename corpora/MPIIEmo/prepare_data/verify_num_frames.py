import os
import sys
import pandas as pd
from imutils.video import count_frames
import cv2
print("Completed imports")
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

''' script verifies that the number of frames in the downsample 10fps videos is
    more or less equal to the number of annotated frames '''

MPIIEMO_DATA_DIR = constants["MPIIEMO_DATA_DIR"]
TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
# load one df of annotations
df = pd.read_csv(os.path.join(MPIIEMO_DATA_DIR, "df0.csv"))
video_id_2_videotime = {}
video_id_2_frames = {}
views = [1,2,3,4,5,6,7,8]
unique_videos = df["video_ids"].unique()
# get number of frames that should be in each video
for video_id in unique_videos:
    video_df = df[df["video_ids"] == video_id]
    video_id_2_videotime[video_id] = video_df["videoTime"].max()
print("videoTimes measured")
# get num frames that are actually in each video
for video_id in unique_videos:
    print("Getting frame counts for {}".format(video_id))
    video_id_2_frames[video_id] = {}
    for view in views:
        video_path = os.path.join(TEN_FPS_VIEWS_DIR, 'view'+str(view), video_id+'.avi')
        video_id_2_frames[video_id][view] = count_frames(video_path)

print(video_id_2_videotime[unique_videos[0]])
# print out results
incorrect_frame_alignment = []
print("VIDEOS WITH CORRECT ALIGNMENT")
for video_id in unique_videos:
    if abs(video_id_2_frames[video_id][1] - video_id_2_videotime[video_id]) > 2:
        incorrect_frame_alignment.append(video_id)
    else:
        print(video_id)
        print("videoTime", video_id_2_videotime[video_id])
        for view, frames in video_id_2_frames[video_id].items():
            print('\t', view, frames)
        print('\n')

print("VIDEOS WITH INCORRECT ALIGMENT")
for video_id in incorrect_frame_alignment:
        print(video_id)
        print("videoTime", video_id_2_videotime[video_id])
        for view, frames in video_id_2_frames[video_id].items():
            print('\t', view, frames)
        print('\n')


