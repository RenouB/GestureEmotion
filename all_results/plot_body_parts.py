import matplotlib.pyplot as plt
import numpy as np
#anger
anger = [[16.9, 17.2, 18.5, 31.0, 28.5],
[13.8, 11.9, 11.1, 30.5, 33.6],
[21.8, 23.7, 18.0,26.9, 31.1],
[17.7, 17.4, 10.2, 22.1, 27.5],
[8.10, 8.05, 0.00, 12.4, 14.3]]
#happiness
happiness = [[18.2, 15.3, 17.4, 32.3, 31.9],
[25.2, 23.2, 15.1, 27.1],
 [20.4, 21.4, 10.1, 25.9, 25.5],
[1.28, 0.00, 0.00, 19.0, 17.0],
[7.78, 5.66, 0.25, 19.1, 16.0]]
#sadness
sadness = [[11.6, 10.8, 11.0, 19.7, 16.2],
[5.46, 6.43, 3.39, 14.9, 11.7],
[9.69, 9.24, 6.04, 11.5, 11.8],
[20.9, 18.7, 4.30, 13.7],
[10.6, 10.9, 0.01, 12.7]]

#surprise
surprise = [[13.3, 15.3, 15.0, 14.2],
[20.5, 18.4, 20.2, 14.5],
[12.3, 11.1, 16.3, 13.6],
[0.08, 0.08, 24.1],
[11.4, 10.3, 10.4, 7.45]]

for emotion in [anger, happiness, sadness, surprise]:
    to_plot = []
    for body_part in emotion:
        mean = np.array(body_part).mean()
        to_plot.append(mean)
    plt.bar(x=[1,2,3,4,5], height=to_plot, width=0.8, tick_label=['full','full-head','full-hh','head','hands'])
    if emotion == anger:
        emotion_title = 'anger'
    if emotion == happiness:
        emotion_title = 'happiness'
    if emotion == sadness:
        emotion_title = 'sadness'
    if emotion == surprise:
        emotion_title = 'surprise'
    plt.title(emotion_title)
    plt.show()
