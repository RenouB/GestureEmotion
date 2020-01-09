import os, sys
import numpy as np
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
import pickle

PROCESSED_BODY_FEATS_DIR = constants["PROCESSED_BODY_FEATS_DIR"]

with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'train-all_manually_selected_cnn.pkl'), 'rb') as f:
	train = pickle.load(f)
with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'test-all_manually_selected_cnn.pkl'), 'rb') as f:
	test = pickle.load(f)
with open(os.path.join(PROCESSED_BODY_FEATS_DIR, 'all_manually_selected_cnn.pkl'), 'rb') as f:
	together = pickle.load(f)

train_counts = {'intact': 0, 'total':0, 'none':0, 'tooclose':0, 'toomany':0, 'nocenter':0}
test_counts = {'intact': 0, 'total':0, 'none':0, 'tooclose':0, 'toomany':0, 'nocenter':0}
together_counts = {'intact': 0, 'total':0, 'none':0, 'tooclose':0, 'toomany':0, 'nocenter':0}

for counts, data in [(train_counts, train), (test_counts, test), (together_counts, together)]:
	for video, views in data.items():
		actorA=video.split('_')[2][1:]
		actorB=video.split('_')[3][1:-4]
		for view, actors in views.items():
			for actor, frames in actors.items():
				for frame, keypoints in frames.items():
					counts['total'] += 1
					
					if type(keypoints) == str:
						counts[keypoints] += 1
						continue
					counts['intact'] += 1
					if actor == 'A':
						if actorA not in counts:
							counts[actorA] = 1
						else:
							counts[actorA] += 1
					elif actor == 'B':
						if actorB not in counts:
							counts[actorB] = 1
						else:
							counts[actorB] += 1

print("######## TRAIN DATA ##############")
print("Total frames: {}".format(train_counts['total']))
print("Percentage with intact keypoints: {:.2f}".format(train_counts['intact'] / train_counts['total']))
print("Total intact: {}".format(train_counts["intact"]))
for key in ['none', 'tooclose', 'toomany', 'nocenter']:
	print(key, "{:.2f}".format(train_counts[key] / train_counts['total']))
for actor in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']:
	print(actor, train_counts[actor])
print("\n")
print("######## TEST DATA ##############")
print("Total frames: {}".format(test_counts['total']))
print("Percentage with intact keypoints: {:.2f}".format(test_counts['intact'] / test_counts['total']))
print("Total intact: {}".format(test_counts["intact"]))
for key in ['none', 'tooclose', 'toomany', 'nocenter']:
	print(key, "{:.2f}".format(test_counts[key] / test_counts['total']))
for actor in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']:
	print(actor, test_counts[actor])
print("\n")
print("######## ALL TOGETHER ##############")
print("Total frames: {}".format(together_counts['total']))
print("Percentage with intact keypoints: {:.2f}".format(together_counts['intact'] / together_counts['total']))
print("Total intact: {}".format(together_counts["intact"]))
for key in ['none', 'tooclose', 'toomany', 'nocenter']:
	print(key, "{:.2f}".format(together_counts[key] / together_counts['total']))
for actor in ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16']:
	print(actor, together_counts[actor])	