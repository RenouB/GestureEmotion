python train_cnn.py -num_folds 8 -emotion 0 -input deltas -keypoints full -epochs 54 -batchsize 20 -cuda -l2 0.001 > anger_full.out &
python train_cnn.py -num_folds 8 -emotion 0 -input deltas -keypoints full-hh -epochs 45 -batchsize 20 -cuda -l2 0.001 > anger_full_hh.out &
python train_cnn.py -num_folds 8 -emotion 0 -input deltas -keypoints full-head -epochs 37 -batchsize 20 -cuda -l2 0.001 > anger_full_head.out &
python train_cnn.py -num_folds 8 -emotion 0 -input deltas -keypoints head -epochs 3 -batchsize 20 -cuda -l2 0.001 > anger_head.out &
python train_cnn.py -num_folds 8 -emotion 0 -input deltas -keypoints hands -epochs 3 -batchsize 20 -cuda -l2 0.001 > anger_hands.out &

python train_cnn.py -num_folds 8 -emotion 1 -input deltas -keypoints full -epochs 37 -batchsize 20 -cuda -l2 0.001 > happiness_full.out &
python train_cnn.py -num_folds 8 -emotion 1 -input deltas -keypoints full-hh -epochs 55 -batchsize 20 -cuda -l2 0.001 > happiness_full_hh.out &
python train_cnn.py -num_folds 8 -emotion 1 -input deltas -keypoints full-head -epochs 34 -batchsize 20 -cuda -l2 0.001 > happiness_full_head.out &
python train_cnn.py -num_folds 8 -emotion 1 -input deltas -keypoints head -epochs 3 -batchsize 20 -cuda -l2 0.001 > happiness_head.out &
python train_cnn.py -num_folds 8 -emotion 1 -input deltas -keypoints hands -epochs 3 -batchsize 20 -cuda -l2 0.001 > happiness_hands.out &

CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 2 -input deltas -keypoints full -epochs 55 -batchsize 20 -cuda -l2 0.001 > sadness_full.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 2 -input deltas -keypoints full-hh -epochs 5 -batchsize 20 -cuda -l2 0.001 > sadness_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 2 -input deltas -keypoints full-head -epochs 2 -batchsize 20 -cuda -l2 0.001 > sadness_full_head.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 2 -input deltas -keypoints head -epochs 3 -batchsize 20 -cuda -l2 0.001 > sadness_head.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 2 -input deltas -keypoints hands -epochs 3 -batchsize 20 -cuda -l2 0.001 > sadness_hands.out &

CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 3 -input deltas -keypoints full -epochs 47 -batchsize 20 -cuda -l2 0.001 > surprise_full.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 3 -input deltas -keypoints full-hh -epochs 3 -batchsize 20 -cuda -l2 0.001 > surprise_full_hh.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 3 -input deltas -keypoints full-head -epochs 3 -batchsize 20 -cuda -l2 0.001 > surprise_full_head.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 3 -input deltas -keypoints head -epochs 3 -batchsize 20 -cuda -l2 0.001 > surprise_head.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -num_folds 8 -emotion 3 -input deltas -keypoints hands -epochs 3 -batchsize 20 -cuda -l2 0.001 > surprise_hands.out &

