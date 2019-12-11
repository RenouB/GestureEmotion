import os, sys
import numpy as np
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
import pickle
import pandas as pd

# what are more questions?
# global and per view: what percentage of frames have at least one person
# what percentage of frames have only one person
# what percentage of frames have two people
# get body keypoint counts normalized by number of skeletons

# do the same thing but per person for both global and view

aggregate = {'num_frames':0, 'skeleton_count':0, 'num_frames_at_least_one_skeleton':0, "num_frames_with_no_skeleton": 0, 
			'num_frames_with_two_skeletons':0, "keypoint_counts":{i:0 for i in range(25)}}
per_view = {}

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]
BODY_KEYPOINT_MAPPING = constants["BODY_KEYPOINT_MAPPING"]

with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'all_raw.pkl'), 'rb') as f:
	all_raw = pickle.load(f)

with open('video_times_and_frames.pkl', 'rb') as f:
	video_times_and_frames = pickle.load(f)

for video, views in list(all_raw.items())[:5]:
	# this assumes that number of frames is same across views, which we verified earlier
	video_id = video[:-4]
	num_frames_in_vid = video_times_and_frames[video_id]["frames_per_view"][1]
	for view in views:
		if view not in per_view:
			per_view[view] = {}
			per_view[view]['num_frames'] = 0
			per_view[view]['num_frames_at_least_one_skeleton'] = 0
			per_view[view]['num_frames_with_no_skeleton'] = 0
			per_view[view]["num_frames_with_two_skeletons"] = 0
		if 'A' not in per_view[view]:
			per_view[view]['A'] = {'skeleton_count': 0, "keypoint_counts": {i:0 for i in range(25)} }
		if 'B' not in per_view[view]:
			per_view[view]['B'] = {'skeleton_count': 0, "keypoint_counts": {i:0 for i in range(25)}}

		
		working_view = per_view[view]
		working_view['num_frames'] += num_frames_in_vid

		A = views[view]['A']
		B = views[view]['B']

		working_view['A']['skeleton_count'] += len(A)
		working_view['B']['skeleton_count'] += len(B)
		working_view['num_frames_with_two_skeletons'] += len([frame for frame in A if frame in B])
		working_view['num_frames_at_least_one_skeleton'] += len(set(A.keys()).union(B.keys()))
		working_view['num_frames_with_no_skeleton'] += len([i for i in range(num_frames_in_vid) \
															if i not in A.keys() and i not in B.keys()])
		
		for frame, keypoints in A.items():
			for i, row in enumerate(keypoints):
				if row.sum() != 0:
					working_view['A']['keypoint_counts'][i] += 1
		
		for frame, keypoints in B.items():
			for i, row in enumerate(keypoints):
				if row.sum() != 0:
					working_view['B']['keypoint_counts'][i] += 1

for view in per_view:
	working_view = per_view[view]
	aggregate['num_frames'] += working_view['num_frames']
	aggregate['skeleton_count'] += working_view['A']['skeleton_count'] + working_view['B']['skeleton_count']
	aggregate['num_frames_with_two_skeletons'] += working_view['num_frames_with_two_skeletons'] 
													
	aggregate['num_frames_at_least_one_skeleton'] += working_view['num_frames_at_least_one_skeleton'] 
	aggregate['num_frames_with_no_skeleton'] += working_view['num_frames_with_no_skeleton'] 

	for i in range(25):
		aggregate["keypoint_counts"][i] += working_view['A']['keypoint_counts'][i]
		aggregate["keypoint_counts"][i] += working_view['B']['keypoint_counts'][i]

	# add together keypoint coutns for A, B
	working_view['all'] = {}
	working_view['all']['skeleton_count'] = working_view['A']['skeleton_count'] + \
												working_view['B']['skeleton_count'] 
	working_view['all']['keypoint_counts'] = {i: working_view['A']["keypoint_counts"][i] + \
													working_view['B']["keypoint_counts"][i] for i in \
													range(25)}
	# normalize numbers for each view
	working_view['A']['keypoint_counts'] = {i: count / working_view['A']['skeleton_count'] for \
												i, count in working_view['A']["keypoint_counts"].items()}
	working_view['B']['keypoint_counts'] = {i: count / working_view['B']['skeleton_count'] for \
												i, count in working_view['B']["keypoint_counts"].items()}
	working_view['all']['keypoint_counts'] = {i : count / working_view['all']['skeleton_count'] for i, count \
													in working_view['all']['keypoint_counts'].items()}

	working_view["num_frames_at_least_one_skeleton"] /= working_view['num_frames']
	working_view["num_frames_with_two_skeletons"] /= working_view['num_frames']
	working_view["num_frames_with_no_skeleton"] /= working_view['num_frames']
# normalize numbers for aggregate
aggregate["keypoint_counts"] = {i: count / aggregate["skeleton_count"] for \
												i, count in aggregate["keypoint_counts"].items()}
aggregate["num_frames_at_least_one_skeleton"] /= aggregate['num_frames']
aggregate["num_frames_with_two_skeletons"] /= aggregate['num_frames']
aggregate["num_frames_with_no_skeleton"] /= aggregate['num_frames']


head_indices = [0, 1, 15, 16, 17, 18]
torso_indices = [2, 3, 4, 5, 6, 7, 8, 9, 12, 19, 20]
legs_indices = [10, 11, 13, 14, 21, 22, 23, 24]
all_together = head_indices + torso_indices + legs_indices

index = ["one_skel", "two_skels", "no_skels", "all_head", "all_torso", "all_legs", "A_head", "A_torso", "A_legs",
		"B_head", "B_torso", "B_legs", "all_mean", "A_mean", "B_mean"] + \
		['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in all_together] + \
		['A_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in all_together] + \
		['B_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in all_together]
# make it all into a df for nice printing
df = pd.DataFrame(index=index, columns=['aggregate','view1','view2','view3','view4','view5','view6','view7','view8'])



df.loc['one_skel', 'aggregate'] = round(aggregate["num_frames_at_least_one_skeleton"], 2)
df.loc['two_skels', 'aggregate'] = round(aggregate["num_frames_with_two_skeletons"], 2)
df.loc['no_skels', 'aggregate'] = round(aggregate["num_frames_with_no_skeleton"], 2)

for i in range(25):
	df.loc['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i], 'aggregate'] = round(aggregate['keypoint_counts'][i]*100)
df.loc['all_head', "aggregate"] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], "aggregate"].mean())
df.loc['all_torso', "aggregate"] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in torso_indices], "aggregate"].mean())
df.loc['all_legs', "aggregate"] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in legs_indices], "aggregate"].mean())
df.loc['all_mean', 'aggregate'] = round(df.loc[[index for index in df.index if index.startswith('all')], 'aggregate'].mean())

for view in per_view:
	working_view = per_view[view]
	df.loc['one_skel', view] = round(working_view["num_frames_at_least_one_skeleton"], 2)
	df.loc['two_skels', view] = round(working_view["num_frames_with_two_skeletons"], 2)
	df.loc['no_skels', view] = round(working_view["num_frames_with_no_skeleton"], 2)

	for i in range(25):
		df.loc['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i], view] = round(working_view['all']['keypoint_counts'][i]*100)
		df.loc['A_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i], view] = round(working_view['A']['keypoint_counts'][i]*100)
		df.loc['B_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i], view] = round(working_view['B']['keypoint_counts'][i]*100)
	
	df.loc['all_head', view] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], view].mean())
	df.loc['all_torso', view] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in torso_indices], view].mean())
	df.loc['all_legs', view] = round(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in legs_indices], view].mean())
	df.loc['A_head', view] = round(df.loc[['A_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], view].mean())
	df.loc['A_torso', view] = round(df.loc[['A_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in torso_indices], view].mean())
	df.loc['A_legs', view] = round(df.loc[['A_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in legs_indices], view].mean())
	df.loc['B_head', view] = round(df.loc[['B_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], view].mean())
	df.loc['B_torso', view] = round(df.loc[['B_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in torso_indices], view].mean())
	df.loc['B_legs', view] = round(df.loc[['B_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in legs_indices], view].mean())
	df.loc['all_mean', view] = round(df.loc[[index for index in df.index if index.startswith('all')], view].mean())
	df.loc['A_mean', view] = round(df.loc[[index for index in df.index if index.startswith('A')], view].mean())
	df.loc['B_mean', view] = round(df.loc[[index for index in df.index if index.startswith('B')], view].mean())


print(df)
# print(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], "view1"].mean())
# print(df.loc[['all_'+str(i)+'_'+BODY_KEYPOINT_MAPPING[i] for i in head_indices], "view2"].mean())