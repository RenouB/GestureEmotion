CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input deltas-noatt -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input deltas-noatt -keypoints full-hh -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input deltas-noatt -keypoints full-head -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input deltas-noatt -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/anger_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input deltas-noatt -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/anger_hands.out &

CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input deltas-noatt -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input deltas-noatt -keypoints full-hh -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input deltas-noatt -keypoints full-head -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input deltas-noatt -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/happiness_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input deltas-noatt -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/happiness_hands.out &

CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input deltas-noatt -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/sadness_full.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input deltas-noatt -keypoints full-hh -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input deltas-noatt -keypoints full-head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_full_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input deltas-noatt -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input deltas-noatt -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_hands.out &

CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input deltas-noatt -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/surprise_full.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input deltas-noatt -keypoints full-hh -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input deltas-noatt -keypoints full-head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_full_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input deltas-noatt -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_head.out &
CUDA_VISIBLE_DEVICES=1 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input deltas-noatt -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_hands.out &

CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input brute -interp -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input brute -interp -keypoints full-hh -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full_hh.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input brute -interp -keypoints full-head -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/anger_full_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input brute -interp -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/anger_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 0 -input brute -interp -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/anger_hands.out &

CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input brute -interp -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input brute -interp -keypoints full-hh -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full_hh.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input brute -interp -keypoints full-head -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/happiness_full_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input brute -interp -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/happiness_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 1 -input brute -interp -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/happiness_hands.out &

CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input brute -interp -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/sadness_full.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input brute -interp -keypoints full-hh -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_full_hh.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input brute -interp -keypoints full-head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_full_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input brute -interp -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 2 -input brute -interp -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/sadness_hands.out &

CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input brute -interp -keypoints full -epochs 100  -batchsize 20 -cuda -l2 0.001 > outs/surprise_full.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input brute -interp -keypoints full-hh -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_full_hh.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input brute -interp -keypoints full-head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_full_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input brute -interp -keypoints head -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_head.out &
CUDA_VISIBLE_DEVICES=2 python3 train_joint_bilstm.py -num_folds 8 -emotion 3 -input brute -interp -keypoints hands -epochs 100 -batchsize 20 -cuda -l2 0.001 > outs/surprise_hands.out &
