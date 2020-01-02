import sys
import os
import numpy as np
from shutil import copyfile
np.random.seed(100)

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

'''
this script will copy a random selection of 5% of the verification
images to a new folder so that they can subsequently be verified
manually
'''

VERIFICATION_IMAGES_DIR = constants["VERIFICATION_IMAGES_DIR"]

# if random_selection subdir doesn't exist, make it
if 'random_selection' not in os.listdir(VERIFICATION_IMAGES_DIR):
    os.mkdir(os.path.join(VERIFICATION_IMAGES_DIR, 'random_selection'))

for view in [view for view in os.listdir(VERIFICATION_IMAGES_DIR) \
    if view != 'random_selection']:
        os.mkdir(os.path.join(VERIFICATION_IMAGES_DIR, 'random_selection', view))
        for video in os.listdir(os.path.join(VERIFICATION_IMAGES_DIR, view)):
            images = os.listdir(os.path.join(VERIFICATION_IMAGES_DIR, view, video))
            randomly_chosen_images = np.random.choice(images, len(images) // 10)
            for image in randomly_chosen_images:
                copyfile(os.path.join(VERIFICATION_IMAGES_DIR, view, video, image), \
                    os.path.join(VERIFICATION_IMAGES_DIR, 'random_selection', view, image))
