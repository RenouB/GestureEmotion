import os, sys
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
import pickle

'''double check to make sure all views have same num frames
and that num jsons output by openpose for a given video matches this number'''

RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]

with open("video_times_and_frames.pkl", 'rb') as f:
    video_times_and_frames = pickle.load(f)

# make sure all views have same num frames
unaligned_frame_counts = []
for video, dct in video_times_and_frames.items():
    if len(set(dct["frames_per_view"].values())) != 1:
        unaligned_frame_counts.append(video)
        print(video, ' '.join([frame_count for frame_count \
            in dct["frames_per_view"].values()]))

if len(unaligned_frame_counts) == 0:
    print("All views frame a single video gave same num frames.")

# check num open pose json outputs against num frames
counted_jsons_per_video = {}
for view_folder in  [folder for folder in os.listdir(RAW_BODY_FEATS_DIR) if folder != "all"]:
    for video in os.listdir(os.path.join(RAW_BODY_FEATS_DIR, view_folder)):
        video_id = video[:-4]
        if video_id not in counted_jsons_per_video:
            counted_jsons_per_video[video_id] = {}

        counted_jsons_per_video[video_id][view_folder] = len(os.listdir(os.path.join(RAW_BODY_FEATS_DIR, view_folder, video)))
        if counted_jsons_per_video[video_id][view_folder] != \
            video_times_and_frames[video_id]["frames_per_view"][int(view_folder[-1])]:
            print("mismatch!")
            print("{} {} {} instead of {}".format(video_id, view_folder, counted_jsons_per_video[video_id][view_folder], video_times_and_frames[video_id]["frames_per_view"][int(view_folder[-1])]))
print("Done checking if JSONS match number of frames. If no errors output, everything is good.")
