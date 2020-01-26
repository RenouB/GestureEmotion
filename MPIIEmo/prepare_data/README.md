This folder contains scripts to process raw Openpose body keypoint data.

For each video, openpose generates keypoints for each frame at a framerate of 10fps.

Each video contains two actors. The keypoints must be associated together to form a sequence of keypoints corresponding to one person. Each of these sequences must be associated to the correct actor label, and then to the correct annotation.

Two major sets of techniques were developed. The first set is contained within spatial-keypoint-alignment-(deprecated). Different spatial criteria, such as similarity between keypoint vectors between frames, were developed. These techniques were not working well and difficult to interpret, and so were abandoned.

visual-keypoint-alignment attempts to solve this problem by using openpose keypoints to crop images of bodies from video frames. Two seperate methods are developed, both using color histograms. 

The first method simply computes the color histogram of the cropped image to be asigned, and compares it to a reference image using different distance metrics.

In the second method, seperate SVMs are trained for each actor pair, using color histograms as input.
