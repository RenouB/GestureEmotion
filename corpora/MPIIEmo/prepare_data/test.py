from process_utils import plot_body_keypoints, convert_body_keypoints_to_tuples
import json

with open("test/json/test_video_000000000005_keypoints.json") as f:
    js = json.load(f)

pose = js['people'][0]['pose_keypoints_2d']
pose_tuples = convert_body_keypoints_to_tuples(pose)
plot_body_keypoints(pose_tuples)
