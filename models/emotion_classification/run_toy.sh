cd JointBiLSTM
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -emotion 0 -debug -num_folds 2 -interp -keypoints full -epochs 2 -cuda
