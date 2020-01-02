import numpy as np
import os, sys
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
# quick fix to get script to run from anywhere
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
import cv2
from definitions import constants

''' contains many utilities used to process raw body features '''

# indices of stable body parts such as shoulder, head, hip, etc.
WAIST_UP_BODY_PART_INDICES = constants["WAIST_UP_BODY_PART_INDICES"]
NECK = constants["NECK"] 
BODY_CENTER = constants["BODY_CENTER"]
WAIST_UP_FILTER_RATIO = constants["WAIST_UP_FILTER_RATIO"]
TOO_CLOSE_THRESHOLD = constants["TOO_CLOSE_THRESHOLD"]
ACTOR_REFERENCE_IMAGES_DIR = constants["ACTOR_REFERENCE_IMAGES_DIR"]


def filter_view6(all_keypoints):
	keypoints = [keypoints for keypoints in all_keypoints if keypoints[BODY_CENTER][1] >= 300 ]
	if len(keypoints) == 0:
		return ['none', 'none']
	else:
		return keypoints

def filter_missing_important(keypoints):
	if type(keypoints) == str:
		return keypoints

	elif sum(keypoints[NECK]) == 0 or \
		sum(keypoints[BODY_CENTER]) == 0:
		return "nocenter"
	else:
		return keypoints

def filter_missing_too_many(keypoints):
	if type(keypoints) == str:
		return keypoints
	
	missing_waist_up = 0
	for i in WAIST_UP_BODY_PART_INDICES:
		if sum(keypoints[i]) == 0:
			missing_waist_up += 1
	
	if missing_waist_up > \
		len(WAIST_UP_BODY_PART_INDICES) * WAIST_UP_FILTER_RATIO:
		return "toomany"
	else:
		return keypoints

def filter_too_close(all_keypoints):
	if len(all_keypoints) != 2:
		return all_keypoints

	keypoints1, keypoints2 = all_keypoints
	
	if type(keypoints1) == str or type(keypoints2) == str:
		return all_keypoints

	one_center_x = keypoints1[BODY_CENTER][0]
	two_center_x = keypoints2[BODY_CENTER][0]

	if abs(one_center_x - two_center_x) < TOO_CLOSE_THRESHOLD:
		return ["tooclose", "tooclose"]
	else:
		return [keypoints1, keypoints2]

def filter_keypoints(all_keypoints, view):
	
	# print("start")
	# print([type(keypoints) for keypoints in all_keypoints])
	
	if view == "view6":
		all_keypoints = filter_view6(all_keypoints)
	# print("view6")
	# print([type(keypoints) for keypoints in all_keypoints])
	
	all_keypoints = [filter_missing_important(keypoints) for keypoints in all_keypoints]
	# print("important")
	# print([type(keypoints) for keypoints in all_keypoints])
	
	all_keypoints = [filter_missing_too_many(keypoints) for keypoints in all_keypoints]
	# print("toomany")
	# print([type(keypoints) for keypoints in all_keypoints])
	
	all_keypoints = filter_too_close(all_keypoints)
	# print("tooclose")
	# print([type(keypoints) for keypoints in all_keypoints])
	if len(all_keypoints) == 1:
		all_keypoints = all_keypoints + ['none']

	return all_keypoints

def filter_occlusions(body_keypoints_sequence1, body_keypoints_sequence2):
	# fuck my life
	pass

def get_frame_image_filename(i):
	i += 1
	i_str = str(i)
	if len(i_str) < 4:
		zeros = '0'*(4 - len(i_str))
		return zeros+i_str+'.png' 

def get_crop_coordinates(body_keypoints):
	if type(body_keypoints) == str or body_keypoints is None:
		return None
	min_x = min([coord for coord in body_keypoints[:,0] if coord != 0])
	max_x = body_keypoints[:,0].max()
	min_y = min([coord for coord in body_keypoints[:,1] if coord != 0])
	max_y = body_keypoints[:,1].max()
	
	return [int(coord) for coord in [min_x, max_x, min_y, max_y]]

def get_torso_coordinates(body_keypoints):
	if type(body_keypoints) == str or body_keypoints is None:
		return None
	min_x = max(0, body_keypoints[BODY_CENTER][0]-40)
	max_x = min(body_keypoints[BODY_CENTER][0]+40, 1600)
	min_y = body_keypoints[NECK][1]
	max_y = body_keypoints[BODY_CENTER][1]
	return [int(coord) for coord in [min_x, max_x, min_y, max_y]]

def write_cropped_image(path, image):
	if image is None:
		return
	else:
		cv2.imwrite(path, image)
		return

def write_cropped_images(cropped_images_dir, frame_index, croppedA, croppedB):
	write_cropped_image(os.path.join(cropped_images_dir, 'A', str(frame_index)+'.png'), croppedA)
	write_cropped_image(os.path.join(cropped_images_dir, 'B', str(frame_index)+'.png'), croppedB)
	return

def convert_keypoints_to_array(body_keypoints):
	keypoints = np.array([[body_keypoints[i], body_keypoints[i+1]] for i in range(0,75,3)])
	# for row in keypoints:
	# 	if row[1] == 1224:
	# 		row[1] = 0
	return keypoints

def crop_and_assign(color, distance, only_hue, keypoints1, keypoints2, frame_image, histA, histB, actorA, actorB):
	
	cropped1 = None
	cropped2 = None

	if type(keypoints1) == str and type(keypoints2) == str:
		return None
	
	if type(keypoints1) != str:			
		min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints1)
		# min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints1)

		cropped1 = frame_image[min_y:max_y, min_x:max_x].copy()

	if type(keypoints2) != str:
		min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints2)
		# min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints2)
		cropped2 = frame_image[min_y:max_y, min_x:max_x].copy()

	if cropped1 is not None:
		cropped = cropped1
		compared_keypoints = 1
		remaining_keypoints = 2

	elif cropped2 is not None:
		cropped = cropped2
		compared_keypoints = 2
		remaining_keypoints = 1

	if distance == 'cor':
		distance = cv2.HISTCMP_CORREL
	elif distance == 'chi':
		distance = cv2.HISTCMP_CHISQR

	if color == 'hsv':
		hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
		if only_hue:
			hist = cv2.calcHist(cropped, [0], None, [255], [0,255])
		else:
			hist = get_all_channels_hist(hsv, 255)	
	else:
		hist = get_all_channels_hist(cropped, 256)
	hist = cv2.normalize(hist, hist)

	similarityA = cv2.compareHist(histA, hist, distance)
	similarityB = cv2.compareHist(histB, hist, distance)

	# plot similarity measures...
	# initialize the results figure
	# fig = plt.figure()
	
 
	# # loop over the results
	# ax = fig.add_subplot(1, 3, 1)
	# ax.set_title('comparison im')
	# plt.imshow(cropped)
	# plt.axis("off")

	# ax = fig.add_subplot(1, 3, 2)
	# ax.set_title("{:.2f}".format(similarityA))
	# plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorA+".png")))
	# plt.axis("off")

	# ax = fig.add_subplot(1, 3, 3)
	# ax.set_title("{:.2f}".format(similarityB))
	# plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorB+".png")))
	# plt.axis("off")

	# plt.show()

	if similarityA < similarityB:
		assignment = {compared_keypoints: 'A', remaining_keypoints:'B'}
		if compared_keypoints == 1:
			croppedA = cropped1
			croppedB = cropped2
		if compared_keypoints == 2:
			croppedA = cropped2
			croppedB = cropped1
		return assignment, croppedA, croppedB

	else:
		assignment =  {compared_keypoints: 'B', remaining_keypoints: 'A'}
		if compared_keypoints == 1:
			croppedB = cropped1
			croppedA = cropped2
		if compared_keypoints == 2:
			croppedB = cropped2
			croppedA = cropped1
		return assignment, croppedA, croppedB

def add_keypoints_to_sequences(all_videos, video, view, frame_index, assignment, keypoints1, keypoints2):
	if assignment[1] == 'A':
		all_videos[video][view]['A'][frame_index] = keypoints1
		all_videos[video][view]['B'][frame_index] = keypoints2
	
	else:
		all_videos[video][view]['B'][frame_index] = keypoints1
		all_videos[video][view]['A'][frame_index] = keypoints2

	return all_videos

def get_all_channels_hist(image, num_bins, max_range):
	hist0 = cv2.calcHist(image, [0], None, [min(max_range, num_bins)], [0,max_range])
	hist1 = cv2.calcHist(image, [1], None, [min(max_range, num_bins)], [0,max_range])
	hist2 = cv2.calcHist(image, [2], None, [min(max_range, num_bins)], [0,max_range])

	return np.concatenate([hist0, hist1, hist2], axis=0)

def construct_reference_histograms(color, only_hue, num_bins):
	reference_hists = {}
	reference_indices = {}
	for image in [file for file in os.listdir(ACTOR_REFERENCE_IMAGES_DIR) if file.endswith('.png')]:
		image_id = image[:2]
		# print(image_id)
		original = cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, image))
		# plt.imshow(original)
		# plt.show()
		if color == 'hsv':
			hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
			
			if only_hue:
				hist = cv2.calcHist(hsv, [0], None, [min(num_bins, 255)], [0,255])
			else:
				hist = get_all_channels_hist(hsv, num_bins, 255)

		elif color =='rgb':
			hist = get_all_channels_hist(original, num_bins, 256)
			

		elif color=='gray':
			gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
			hist = cv2.calcHist(gray, [0], None, [num_bins], [0,256])
			
		reference_hists[image_id] = hist
		reference_indices[image_id] = [i for i in range(len(hist))]
	return reference_hists, reference_indices


def construct_pairwise_reference_histograms(color, only_hue, num_bins):
	reference_hists = {}
	reference_indices = {}

	for i in range(1,17,2):
		first_actor = ''+'0'*(2-len(str(i))) + str(i)
		second_actor = ''+'0'*(2-len(str(i+1))) + str(i+1)
	
		first = cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, first_actor+'.png'))
		second = cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, second_actor+'.png'))

		if color == 'hsv':
			first = cv2.cvtColor(first, cv2.COLOR_BGR2HSV)
			second = cv2.cvtColor(second, cv2.COLOR_BGR2HSV)
			
			if only_hue:
				first_hist = cv2.calcHist(first, [0], None, [min(num_bins, 255)], [0,255])
				second_hist = cv2.calcHist(second, [0], None, [min(num_bins, 255)], [0,255])
			else:
				first_hist = get_all_channels_hist(first, num_bins, 255)
				second_hist = get_all_channels_hist(second, num_bins, 255)

		elif color =='rgb':
			first_hist = get_all_channels_hist(first, num_bins, 256)
			second_hist = get_all_channels_hist(second, num_bins, 256)

		elif color=='gray':
			first = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
			second = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
			first_hist = cv2.calcHist(first, [0], None, [num_bins], [0,256])
			second_hist = cv2.calcHist(second, [0], None, [num_bins], [0,256])

		first_hist = cv2.normalize(first_hist, first_hist).flatten()
		second_hist = cv2.normalize(second_hist, second_hist).flatten()

		only_in_first = [i for i in range(len(first_hist)) if first_hist[i] > 0 and second_hist[i] == 0]
		only_in_second = [i for i in range(len(second_hist)) if second_hist[i] > 0 and first_hist[i] == 0]
		indices_to_compare = sorted(only_in_first + only_in_second)

		first_hist = np.array([first_hist[i] for i in indices_to_compare])
		second_hist = np.array([second_hist[i] for i in indices_to_compare])

		reference_hists[first_actor] = first_hist
		reference_hists[second_actor] = second_hist
		reference_indices[first_actor] = indices_to_compare
		reference_indices[second_actor] = indices_to_compare
	
	return reference_hists, reference_indices
