

RAW_BODY_FEATS_DIR = constants["RAW_BODY_FEATS_DIR"]

'''
this script will verify integrity of json file output by openpose
'''

with open("video_times_and_frames.pkl", 'rb') as f:
    video_times_and_frames = pickle.load(f)

problematic_videos = []
for view in os.listdir(RAW_BODY_FEATS_DIR):
    view_folder = os.path.join(RAW_BODY_FEATS_DIR, view)
    for video in os.listdir(view_folder):
        video_id = video[:-4]
        view_as_int = int(view[-1])
        num_jsons = len(os.listdir(os.path.join(view_folder, video)))
        num_frames = video_times_and_frames[video_id]["frames_per_view"][view_as_int]
        if num_jsons != num_frames:
            problematic_videos.append((video_id, view, num_jsons, num_frames))

if len (problematic_videos) == 0:
    print("All videos look good")
else:
    for video in problematic_videos:
        print("{} {} num jsons {} num frames {}".format(video))
