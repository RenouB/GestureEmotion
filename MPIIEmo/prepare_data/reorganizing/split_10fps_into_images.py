import os, sys
PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-4])
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
		os.system("[ -d \"{}\" ] && rm -rf {} \n".format(video_images_dir, video_images_dir))
		os.system("mkdir {}".format(video_images_dir))
		cmd = "ffmpeg -i {} {}/%04d.png -hide_banner -loglevel panic".format(video_path, video_images_dir)
		print(cmd)
		os.system(cmd)
		
 