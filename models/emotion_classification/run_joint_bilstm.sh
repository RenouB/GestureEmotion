find JointBiLSTM -name *pkl -exec rm {} \;
cd JointBiLSTM
CUDA_VISIBLE_DEVICES=3 python3 train_joint_bilstm.py -emotion 0 -cuda -num_folds 8 -epochs 100 -interp -keypoints full -input brute -debug &
CUDA_VISIBLE_DEVICES=3 python3 train_joint_bilstm.py -emotion 1 -cuda -num_folds 8 -epochs 100 -interp -keypoints full -input brute -debug &
CUDA_VISIBLE_DEVICES=3 python3 train_joint_bilstm.py -emotion 2 -cuda -num_folds 8 -epochs 100 -interp -keypoints full -input brute -debug &
CUDA_VISIBLE_DEVICES=3 python3 train_joint_bilstm.py -emotion 3 -cuda -num_folds 8 -epochs 100 -interp -keypoints full -input brute -debug &
