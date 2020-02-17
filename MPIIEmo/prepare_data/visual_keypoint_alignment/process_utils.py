import numpy as np
import os, sys
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt
# quick fix to get script to run from anywhere
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
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

def convert_keypoints_to_array(body_keypoints):
	keypoints = np.array([[body_keypoints[i], body_keypoints[i+1]] for i in range(0,75,3)])
	# for row in keypoints:
	# 	if row[1] == 1224:
	# 		row[1] = 0
	return keypoints

def add_keypoints_to_sequences(all_videos, video, view, frame_index, assignment, keypoints1, keypoints2):
	if assignment[1] == 'A':
		all_videos[video][view]['A'][frame_index] = keypoints1
		all_videos[video][view]['B'][frame_index] = keypoints2

	else:
		all_videos[video][view]['B'][frame_index] = keypoints1
		all_videos[video][view]['A'][frame_index] = keypoints2

	return all_videos

def translate_keypoints(body_keypoints):
	mid_hip = body_keypoints[BODY_CENTER]
	translated_keypoints = []
	for keypoint in body_keypoints:
		if keypoint.sum() == 0:
			translated_keypoints.append(keypoint)
		else:
			translated_keypoints.append(keypoint - mid_hip)

	return np.array(translated_keypoints)

def interpolate_missing_coordinates(one_coord_across_all_frames):
	invalid_coords = [i for i, coords in enumerate(one_coord_across_all_frames)
						if coords.sum() == 0]


	if len(invalid_coords) == len(one_coord_across_all_frames):
		one_coord_across_all_frames = np.array([[2,2]]*len(one_coord_across_all_frames))
		return one_coord_across_all_frames
	for i in invalid_coords:
		valid_coords = np.array([i for i, coords in enumerate(one_coord_across_all_frames)
						if coords.sum() != 0])
		distances_from_valid = abs(valid_coords - i)
		min_distance_index = np.argmin(distances_from_valid)
		one_coord_across_all_frames[i] = one_coord_across_all_frames[valid_coords[min_distance_index]]
	return one_coord_across_all_frames

def interpolate_keypoints_all_frames(keypoints_from_all_frames):
	# keypoint dim = #frames * #bodykeypoints * 2

	keypoints_from_all_frames = np.swapaxes(keypoints_from_all_frames, 1, 2)

	for coord_index in range(keypoints_from_all_frames.shape[2]):
		if coord_index != constants["BODY_CENTER"]:

			keypoints_from_all_frames[:,:,coord_index] = \
				interpolate_missing_coordinates(keypoints_from_all_frames[:,:,coord_index])
	return np.swapaxes(keypoints_from_all_frames,1,2)

def scale_keypoints(body_keypoints):
	neck = body_keypoints[NECK]
	mid_hip = body_keypoints[BODY_CENTER]
	l2 = np.linalg.norm(neck - mid_hip)

	if l2 == 0:
		return body_keypoints
	return body_keypoints / l2

def normalize_keypoints(keypoints):
	if type(keypoints) == str:
		return keypoints
	return scale_keypoints(translate_keypoints(keypoints))
############# Filtering

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
	one_neck_x = keypoints1[NECK][0]
	two_neck_x = keypoints2[NECK][0]
	if abs(one_center_x - two_center_x) < TOO_CLOSE_THRESHOLD or \
		abs(one_neck_x - two_neck_x) < TOO_CLOSE_THRESHOLD:
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
	all_keypoints = filter_too_close(all_keypoints)

	all_keypoints = [filter_missing_important(keypoints) for keypoints in all_keypoints]
	# print("important")
	# print([type(keypoints) for keypoints in all_keypoints])

	all_keypoints = [filter_missing_too_many(keypoints) for keypoints in all_keypoints]
	# print("toomany")
	# print([type(keypoints) for keypoints in all_keypoints])

	# print("tooclose")
	# print([type(keypoints) for keypoints in all_keypoints])
	if len(all_keypoints) == 1:
		all_keypoints = all_keypoints + ['none']

	return all_keypoints

def filter_occlusions(body_keypoints_sequence1, body_keypoints_sequence2):
	# fuck my life
	pass


############## IMAGE PROCESSING

def get_frame_image_filename(i):
	i += 1
	i_str = str(i)
	if len(i_str) < 4:
		zeros = '0'*(4 - len(i_str))
		return zeros+i_str+'.jpg'

def get_body_image_filename(i, ii):
	# i += 1
	i_str = str(i)
	ii_str = str(ii)

	return i_str+'-'+ii_str+'.jpg'

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
	write_cropped_image(os.path.join(cropped_images_dir, 'A', str(frame_index)+'.jpg'), croppedA)
	write_cropped_image(os.path.join(cropped_images_dir, 'B', str(frame_index)+'.jpg'), croppedB)
	return


def write_cropped_images_actor_irrelevant(cropped_images_dir, frame_index, cropped1, cropped2):
	write_cropped_image(os.path.join(cropped_images_dir, str(frame_index)+'-1.jpg'), cropped1)
	write_cropped_image(os.path.join(cropped_images_dir, str(frame_index)+'-2.jpg'), cropped2)
	return

def crop(keypoints1, keypoints2, frame_image, only_torsos):

	cropped1 = None
	cropped2 = None


	if type(keypoints1) == str and type(keypoints2) == str:
		return None

	if type(keypoints1) != str:
		if only_torsos:
			min_x, max_x, min_y, max_y = get_torso_coordinates(keypoints1)
		else:
			min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints1)


		cropped1 = frame_image[min_y:max_y, min_x:max_x].copy()

	if type(keypoints2) != str:
		if only_torsos:
			min_x, max_x, min_y, max_y = get_torso_coordinates(keypoints2)
		else:
			min_x, max_x, min_y, max_y = get_crop_coordinates(keypoints2)

		cropped2 = frame_image[min_y:max_y, min_x:max_x].copy()

	# fig = plt.figure()
	# ax = fig.add_subplot(1, 2, 1)
	# if cropped1 is not None:
	# 	plt.imshow(cropped1)
	# else:
	# 	plt.imshow([[0]])
	# ax = fig.add_subplot(1, 2, 2)
	# if cropped2 is not None:
	# 	plt.imshow(cropped2)
	# else:
	# 	plt.imshow([[0]])
	# plt.show()
	return cropped1, cropped2


################ HISTOGRAMS
def convert_colors(cropped1, cropped2, color):
	cropped1_rgb = None
	cropped2_rgb = None
	if cropped1 is not None:
		cropped1_rgb = cropped1.copy()
		if color == 'hsv':
			cropped1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2HSV)
		elif color == 'gray':
			cropped1 = cv2.cvtColor(cropped1, cv2.COLOR_BGR2GRAY)

	if cropped2 is not None:
		cropped2_rgb = cropped2.copy()
		if color == 'hsv':
			cropped2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2HSV)
		elif color == 'gray':
			cropped2 = cv2.cvtColor(cropped2, cv2.COLOR_BGR2GRAY)

	return cropped1_rgb, cropped2_rgb, cropped1, cropped2

def convert_to_hist(cropped1, cropped2, num_bins, only_hue, comparison_indices):
	if cropped1 is not None:
		if only_hue:
			hist1 = cv2.calcHist(cropped1, [0], None, [num_bins], [0,255])
			hist1 = np.array([hist1[i] for i in comparison_indices])
			hist1 = cv2.normalize(hist1, hist1)
		else:
			hist1 = get_all_channels_hist(cropped1, num_bins, 255)
			hist1 = np.array([hist2[i] for i in comparison_indices])
			hist1 = cv2.normalize(hist2, hist2)
	else:
		hist1 = None

	if cropped2 is not None:
		if only_hue:
			hist2 = cv2.calcHist(cropped2, [0], None, [num_bins], [0,255])
			hist2 = np.array([hist2[i] for i in comparison_indices])
			hist2 = cv2.normalize(hist2, hist2)
		else:
			hist2 = get_all_channels_hist(cropped2, num_bins, 255)
			hist2 = np.array([hist2[i] for i in comparison_indices])
			hist2 = cv2.normalize(hist2, hist2)
	else:
		hist2 = None

	return hist1, hist2

def get_most_similar_pair(one_sim_A, one_sim_B, two_sim_A, two_sim_B, reverse):
	all_sims = [one_sim_A, one_sim_B, two_sim_A, two_sim_B]
	for i, sim in enumerate(all_sims):
		if sim is None:
			if reverse:
				all_sims[i] = 100000
			else:
				all_sims[i] = -100000
	if reverse:
		_, best = min([(sim, i) for i, sim in enumerate(all_sims)])
	else:
		_, best = max([(sim, i) for i, sim in enumerate(all_sims)])

	return best

def assign(cropped1, cropped2, histA, histB, color, distance, only_hue,
	num_bins, comparison_indices, actorA, actorB):

	if distance == 'cor':
		distance = cv2.HISTCMP_CORREL
		reverse = False
	elif distance == 'chi':
		distance = cv2.HISTCMP_CHISQR
		reverse = True
	elif distance == 'intersection':
		distance = dv2.HISTCMP_INTERSECTION
		reverse = False
	elif distance == 'euc':
		reverse = True
	cropped1_rgb, cropped2_rgb, cropped1, cropped2 =  \
		convert_colors(cropped1, cropped2, color)
	hist1, hist2 = convert_to_hist(cropped1, cropped2, num_bins, only_hue, comparison_indices)

	if hist1 is not None:
		if distance == 'euc':
			one_sim_A = euclidean(hist1, histA)
			one_sim_B = euclidean(hist1, histB)
		else:
			one_sim_A = cv2.compareHist(histA, hist1, distance)
			one_sim_B = cv2.compareHist(histB, hist1, distance)
	else:
		one_sim_A = None
		one_sim_B = None

	if hist2 is not None:
		if distance == 'euc':
			two_sim_A = euclidean(hist2, histA)
			two_sim_B = euclidean(hist2, histB)
		else:
			two_sim_A = cv2.compareHist(histA, hist2, distance)
			two_sim_B = cv2.compareHist(histB, hist2, distance)
	else:
		two_sim_A = None
		two_sim_B = None

	print("############")
	print("skeleton one")
	print(hist1)
	print("skeleton two")
	print(hist2)
	print("histA")
	print(histA)
	print("histB")
	print(histB)

	fig = plt.figure()

	ax = fig.add_subplot(2, 3, 1)
	ax.set_title('first')
	if cropped1_rgb is not None:
		plt.imshow(cropped1_rgb)
	else:
		plt.imshow([[0]])
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 2)
	ax.set_title("{}".format(str(one_sim_A)))
	plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorA+'.png')))
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 3)
	ax.set_title("{}".format(str(one_sim_B)))
	plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorB+'.png')))
	plt.axis("off")


	ax = fig.add_subplot(2, 3, 4)
	ax.set_title('second')
	if cropped2_rgb is not None:
		plt.imshow(cropped2_rgb)
	else:
		plt.imshow([[0]])
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 5)
	ax.set_title("{}".format(str(two_sim_A)))
	plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorA+'.png')))
	plt.axis("off")

	ax = fig.add_subplot(2, 3, 6)
	ax.set_title("{}".format(str(two_sim_B)))
	plt.imshow(cv2.imread(os.path.join(ACTOR_REFERENCE_IMAGES_DIR, actorB+'.png')))
	plt.axis("off")

	plt.show()
	best = get_most_similar_pair(one_sim_A, one_sim_B, two_sim_A, two_sim_B, reverse)

	if best == 0 or best == 2:
		assignment = {1:'A', 2:'B'}
		croppedA = 0
		croppedB = 1
		# croppedA = cropped1_rgb
		# croppedB = cropped2_rgb
	else:
		assignment = {1:'B', 2:'A'}
		croppedA = 1
		croppedB = 0

	return assignment, croppedA, croppedB


def get_all_channels_hist(image, num_bins, max_range):
	hist0 = cv2.calcHist(image, [0], None, [min(max_range, num_bins)], [0,max_range])
	hist1 = cv2.calcHist(image, [1], None, [min(max_range, num_bins)], [0,max_range])
	hist2 = cv2.calcHist(image, [2], None, [min(max_range, num_bins)], [0,max_range])

	return np.concatenate([hist0, hist1, hist2], axis=0)

def construct_reference_histograms(color, only_hue, num_bins, hist_diff):
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
		if hist_diff:
			indices_to_compare = sorted(only_in_first + only_in_second)
		else:
			indices_to_compare = range(0, len(first_hist))

		first_hist = np.array([first_hist[i] for i in indices_to_compare])
		second_hist = np.array([second_hist[i] for i in indices_to_compare])

		reference_hists[first_actor] = first_hist
		reference_hists[second_actor] = second_hist
		reference_indices[first_actor] = indices_to_compare
		reference_indices[second_actor] = indices_to_compare

	return reference_hists, reference_indices
