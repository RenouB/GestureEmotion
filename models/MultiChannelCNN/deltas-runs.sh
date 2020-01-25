python train_cnn.py -input deltas -emotion 0 -epochs 60 -cuda -keypoints full  > outs/anger-brute-full.out &
python train_cnn.py -input deltas -emotion 0 -epochs 60 -cuda -keypoints full-hh  > outs/anger-brute-full-hh.out &
python train_cnn.py -input deltas -emotion 0 -epochs 60 -cuda -keypoints full-head  > outs/anger-brute-full-head.out &
python train_cnn.py -input deltas -emotion 0 -epochs 60 -cuda -keypoints hands  > outs/anger-brute-hands.out &
python train_cnn.py -input deltas -emotion 0 -epochs 60 -cuda -keypoints head  > outs/anger-brute-head.out &

python train_cnn.py -input deltas -emotion 1 -epochs 60 -cuda -keypoints full  > outs/happiness-brute-full.out &
python train_cnn.py -input deltas -emotion 1 -epochs 60 -cuda -keypoints full-hh  > outs/happiness-brute-full-hh.out &
python train_cnn.py -input deltas -emotion 1 -epochs 60 -cuda -keypoints full-head  > outs/happiness-brute-full-head.out &
python train_cnn.py -input deltas -emotion 1 -epochs 60 -cuda -keypoints hands  > outs/happiness-brute-hands.out &
python train_cnn.py -input deltas -emotion 1 -epochs 60 -cuda -keypoints head  > outs/happiness-brute-head.out &

python train_cnn.py -input deltas -emotion 2 -epochs 60 -cuda -keypoints full  > outs/sadness-brute-full.out &
python train_cnn.py -input deltas -emotion 2 -epochs 60 -cuda -keypoints full-hh  > outs/sadness-brute-full-hh.out &
python train_cnn.py -input deltas -emotion 2 -epochs 60 -cuda -keypoints full-head  > outs/sadness-brute-full-head.out &
python train_cnn.py -input deltas -emotion 2 -epochs 60 -cuda -keypoints hands  > outs/sadness-brute-hands.out &
python train_cnn.py -input deltas -emotion 2 -epochs 60 -cuda -keypoints head  > outs/sadness-brute-head.out &


CUDA_VISIBLE_DEVICES=1 python train_cnn.py -input deltas -emotion 3 -epochs 60 -cuda -keypoints full  > outs/surprise-brute-full.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -input deltas -emotion 3 -epochs 60 -cuda -keypoints full-hh  > outs/surprise-brute-full-hh.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -input deltas -emotion 3 -epochs 60 -cuda -keypoints full-head  > outs/surprise-brute-full-head.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -input deltas -emotion 3 -epochs 60 -cuda -keypoints hands  > outs/surprise-brute-hands.out &
CUDA_VISIBLE_DEVICES=1 python train_cnn.py -input deltas -emotion 3 -epochs 60 -cuda -keypoints head  > outs/surprise-brute-head.out &
