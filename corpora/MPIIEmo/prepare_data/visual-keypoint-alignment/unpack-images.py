import sys
import os
import re
import shutil

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

'''extract downloaded frame images from .zip files and move them to correct
subfolder in 10fps views directory'''

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
DRIVE_DOWNLOAD_DIR = os.path.join(PROJECT_DIR, 'drive_download')

view_re = re.compile("view.*zip")
for file in os.listdir(DRIVE_DOWNLOAD_DIR):
	print(file)
	if view_re.match(file):
		os.system("unzip {} -d {}".format(os.path.join(DRIVE_DOWNLOAD_DIR, file), DRIVE_DOWNLOAD_DIR))

view_re = re.compile("view.*images")
image_subfolders = [f for f in os.scandir(DRIVE_DOWNLOAD_DIR) if f.is_dir() and view_re.match(f.name)]
for subfolder in image_subfolders:
	view = subfolder.name.split('-')[0]
	for video in os.scandir(os.path.join(DRIVE_DOWNLOAD_DIR, subfolder)):
		destination = os.path.join(TEN_FPS_VIEWS_DIR, view, 'images', video.name)
		os.system("rm -rf {}".format(destination))
		os.system("mkdir -p {}".format(destination))
		os.system("cp {}/* {}".format(video.path, destination))
		shutil.rmtree(subfolder.name)