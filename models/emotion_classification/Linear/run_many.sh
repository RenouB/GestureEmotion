CUDA_VISIBLE_DEVICES=5 python3 train_linear.py -interp -emotion 0 -epochs 70 > outs/anger-interp.out &
CUDA_VISIBLE_DEVICES=5 python3 train_linear.py -interp -emotion 1 -epochs 70 > outs/happiness-interp.out &
CUDA_VISIBLE_DEVICES=5 python3 train_linear.py -interp -emotion 2 -epochs 70 > outs/sadness-interp.out &
CUDA_VISIBLE_DEVICES=5 python3 train_linear.py -interp -emotion 3 -epochs 70 > outs/surprise-interp.out &
