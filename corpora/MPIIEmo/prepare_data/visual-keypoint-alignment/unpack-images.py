import sys
import os
import re

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants

TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]
DRIVE_DOWNLOAD_DIR = os.path.join(PROJECT_DIR, 'drive_download')

view_re = re.compile("view.*zip")


for file in os.listdir(DRIVE_DOWNLOAD_DIR):
	print(file)
	if view_re.match(file):
		os.system("unzip {} -d {}".format(os.path.join(DRIVE_DOWNLOAD_DIR, file), DRIVE_DOWNLOAD_DIR))



