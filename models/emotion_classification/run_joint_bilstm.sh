cd JointBiLSTM
CUDA_VISIBLE_DEVICES= python3 train_joint_bilstm.py -emotion 0 -cuda -num_folds 2 -epochs 2 -interp -keypoints full -input brute -debug 
