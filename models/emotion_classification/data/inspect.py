import pickle

with open("interp-perturb-joint-int3-seq5.pkl","rb") as f:
    all = pickle.load(f)

totalA = 0
totalB= 0
for video, views in all.items():
    for view, actors in views.items():
        for actor, frames in actors.items():
            if actor == 'A':
                print(frames['poses'][0][0])
