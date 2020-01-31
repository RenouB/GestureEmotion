CUDA_VISIBLE_DEVICES=1 python train_linear.py -interp -emotion 0 -epochs 100 > outs/anger-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py  -emotion 0 -epochs 100 > outs/anger.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -interp -emotion 1 -epochs 100 > outs/happiness-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -emotion 1 -epochs 100 > outs/happiness-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -interp -emotion 2 -epochs 100 > outs/sadness-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -emotion 2 -epochs 100 > outs/sadness-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -interp -emotion 3 -epochs 100 > outs/surprise-interp.out &
CUDA_VISIBLE_DEVICES=1 python train_linear.py -emotion 3 -epochs 100 > outs/surprise-interp.out &
