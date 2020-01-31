python train_svm.py -kernel sigmoid -C 1 -interp -emotion 0 > outs/anger_interp.out &
python train_svm.py -kernel sigmoid -C 1 -emotion 0 > outs/anger.out &
python train_svm.py -kernel sigmoid -C 1 -interp -emotion 1 > outs/happiness_interp.out &
python train_svm.py -kernel sigmoid -C 1 -emotion 1 > outs/happiness.out &
python train_svm.py -kernel sigmoid -C 1 -interp -emotion 2 > outs/sadness_interp.out &
python train_svm.py -kernel sigmoid -C 1 -emotion 2 > outs/sadness.out &
python train_svm.py -kernel sigmoid -C 1 -interp -emotion 3 > outs/surprise_interp.out &
python train_svm.py -kernel sigmoid -C 1 -emotion 3 > outs/surprise.out &
