import os, sys
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-3])
sys.path.insert(0, PROJECT_DIR)

'''split each 10fps video into individual frames'''

from definitions import constants
TEN_FPS_VIEWS_DIR = constants["TEN_FPS_VIEWS_DIR"]

for view in os.listdir(TEN_FPS_VIEWS_DIR):
	view_dir = os.path.join(TEN_FPS_VIEWS_DIR, view)
	images_dir = os.path.join(view_dir, 'images')
	os.system("mkdir {}".format(images_dir))
	for video in os.listdir(view_dir):
		video_path = os.path.join(view_dir, video)
		video_images_dir = os.path.join(images_dir, video)
		os.system("mkdir {}".format(video_images_dir))
		print("before ffmpeg")
		cmd = "ffmpeg -i {} {}/%04d.png -hide_banner".format(video_path, video_images_dir)
		print(cmd)
		os.system(cmd)
 