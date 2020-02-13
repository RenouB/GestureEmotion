# Body keypoint features for emotion recognition

The work in this repo was developed during the Fachpraktikum HCI and Machine Learning
for Computer Vision at the University of Stuttgart, Winter 2019/2020 semester.

A variety of machine learning and neural network models have been implemented to
predict the discrete emotion categories anger, sadness, surprise and sadness using
only body keypoint features extracted using the open source framework openpose.

All emotion classification models can be found in models/emotion_classification.

Additionally, four models were developed to address the pre-processing task of pose alignment. These can be found in models/visual_pose_alignment.

In addition to the code for the models, which have been prepared for public use,
this repo also contains many data preprocessing scripts in MPIIEmo/prepare_data.
Most of these will not work if run (the data files required are not provided),
but should the reader decide to work directly with the MPIIEmo data, these scripts
may prove useful.

## SETUP INSTRUCTIONS:

Download emotion classification data from
https://drive.google.com/open?id=1Iy77VZ1MXNa3qU7h4YLj8MkdMNLqAHMy

Unzip into models/emotion_classification/data.

Download pose alignment data from
https://drive.google.com/open?id=1vcXJz4zeCrVkS7W5ahsTLlpNznPsmF8c

Unzip into models/pose_alignment/data.

create your virtual environment and ensure you have all requirements in
requirements.txt

## HOW TO RUN?
Navigate to the folder of a given model (for example, models/emotion_classification/BiLSTM).
Here you will find a readme with example commands that may be run to train and test the model (hopefully reproducing reported results).

## WHERE ARE MY OUTPUTS?
Most models have an analogous directory structure. At it's deepest point, (for example, BiLSTM/brute/anger/ind/pose/) there will be logs, weights and scores directories. All outputs will be stored there.
