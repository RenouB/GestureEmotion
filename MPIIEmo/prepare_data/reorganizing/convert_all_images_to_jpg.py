import os
import sys
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]

for folder in os.scandir(TEN_FPS_VIEWS_DIR):
	print(folder)
	images_folder = os.path.join(folder.path, 'images')
	i = 0
	for video in os.scandir(images_folder):
		i += 1

		print(i)
		print(video.path)
		os.system("magick mogrify -format jpg {}/*.png".format(video.path))
		os.system("rm {}/*.png".format(video.path))