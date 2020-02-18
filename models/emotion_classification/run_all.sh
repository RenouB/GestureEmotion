# run random
cd rand
python3 -u rand.py -emotion 0  > outs/rand0.out
python3 -u rand.py -emotion 2  > outs/rand1.out
python3 -u rand.py -emotion 2  > outs/rand2.out
python3 -u rand.py -emotion 3  > outs/rand3.out

# run SVM
cd ../SVM
python3 -u train_svm.py -emotion 0 > outs/svm0.out
python3 -u train_svm.py -emotion 1 > outs/svm1.out
python3 -u train_svm.py -emotion 2 > outs/svm2.out
python3 -u train_svm.py -emotion 3 > outs/svm3.out

# run Linear
cd ../Linear
CUDA_VISIBLE_DEVICES=2 python3 -u train_linear.py -epochs 20 -interp -cuda -emotion 0 > outs/linear0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_linear.py -epochs 20 -interp -cuda -emotion 1 > outs/linear1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_linear.py -epochs 20 -interp -cuda -emotion 2 > outs/linear2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_linear.py -epochs 20 -interp -cuda -emotion 3 > outs/linear3.out

# run CNN
cd ../MultiChannelCNN
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full > outs/cnn-full0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full > outs/cnn-full1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full > outs/cnn-full2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full > outs/cnn-full3.out

CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full-head > outs/cnn-full-head0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full-head > outs/cnn-full-head1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full-head > outs/cnn-full-head2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full-head > outs/cnn-full-head3.out

CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full-hh > outs/cnn-full-hh0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full-hh > outs/cnn-full-hh2.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full-hh > outs/cnn-full-hh1.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full-hh > outs/cnn-full-hh3.out

CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  head > outs/cnn-head0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  head > outs/cnn-head1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  head > outs/cnn-head2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  head > outs/cnn-head3.out

CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  hands > outs/cnn-hands0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  hands > outs/cnn-hands1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  hands > outs/cnn-hands2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  hands > outs/cnn-hands3.out

#run attCNN
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input deltas -cuda -keypoints  full > outs/att-cnn-full0.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input deltas -cuda -keypoints  full > outs/att-cnn-full1.out
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input deltas -cuda -keypoints  full > outs/att-cnn-full2.out &
CUDA_VISIBLE_DEVICES=2 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input deltas -cuda -keypoints  full > outs/att-cnn-full3.out

CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input deltas -cuda -keypoints  full-head > outs/att-cnn-full-head0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input deltas -cuda -keypoints  full-head > outs/att-cnn-full-head1.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input deltas -cuda -keypoints  full-head > outs/att-cnn-full-head2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input deltas -cuda -keypoints  full-head > outs/att-cnn-full-head3.out &

CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input deltas -cuda -keypoints  full-hh > outs/att-cnn-full-hh0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input deltas -cuda -keypoints  full-hh > outs/att-cnn-full-hh1.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input deltas -cuda -keypoints  full-hh > outs/att-cnn-full-hh2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input deltas -cuda -keypoints  full-hh > outs/att-cnn-full-hh3.out &

CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input deltas -cuda -keypoints  head > outs/att-cnn-head0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input deltas -cuda -keypoints  head > outs/att-cnn-head1.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input deltas -cuda -keypoints  head > outs/att-cnn-head2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input deltas -cuda -keypoints  head > outs/att-cnn-head3.out &

CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 0 -input deltas -cuda -keypoints  hands > outs/att-cnn-hands0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 1 -input deltas -cuda -keypoints  hands > outs/att-cnn-hands1.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 2 -input deltas -cuda -keypoints  hands > outs/att-cnn-hands2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_cnn.py -epochs 20 -interp -emotion 3 -input deltas -cuda -keypoints  hands > outs/att-cnn-hands3.out &

# run BiLSTM brute
cd ../BiLSTM
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full > outs/bilstm-full0-brute.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full > outs/bilstm-full1-brute.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full > outs/bilstm-full2-brute.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full > outs/bilstm-full3-brute.out

CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full-head > outs/bilstm-full--brutehead0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full-head > outs/bilstm-full--brutehead1.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full-head > outs/bilstm-full--brutehead2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full-head > outs/bilstm-full--brutehead3.out

CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  full-hh > outs/bilstm-full--brutehh0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  full-hh > outs/bilstm-full--brutehh1.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  full-hh > outs/bilstm-full--brutehh2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  full-hh > outs/bilstm-full--brutehh3.out

CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  head > outs/bilstm-head0-brute.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  head > outs/bilstm-head1-brute.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  head > outs/bilstm-head2-brute.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  head > outs/bilstm-head3-brute.out

CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input brute -cuda -keypoints  hands > outs/bilstm-hands-brute0.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input brute -cuda -keypoints  hands > outs/bilstm-hands-brute1.out
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input brute -cuda -keypoints  hands > outs/bilstm-hands-brute2.out &
CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input brute -cuda -keypoints  hands > outs/bilstm-hands-brute3.out

# # run BiLSTM delta
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -cuda -keypoints  full > outs/bilstm-full0-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -cuda -keypoints  full > outs/bilstm-full1-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -cuda -keypoints  full > outs/bilstm-full2-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -cuda -keypoints  full > outs/bilstm-full3-deltas-noatt.out
#
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -cuda -keypoints  full-head > outs/bilstm-full--deltas-noatthead0.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -cuda -keypoints  full-head > outs/bilstm-full--deltas-noatthead1.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -cuda -keypoints  full-head > outs/bilstm-full--deltas-noatthead3.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -cuda -keypoints  full-head > outs/bilstm-full--deltas-noatthead2.out
#
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -cuda -keypoints  full-hh > outs/bilstm-full--deltas-noatthh0.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -cuda -keypoints  full-hh > outs/bilstm-full--deltas-noatthh1.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -cuda -keypoints  full-hh > outs/bilstm-full--deltas-noatthh2.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -cuda -keypoints  full-hh > outs/bilstm-full--deltas-noatthh3.out
#
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -cuda -keypoints  head > outs/bilstm-head0-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -cuda -keypoints  head > outs/bilstm-head1-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -cuda -keypoints  head > outs/bilstm-head2-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -cuda -keypoints  head > outs/bilstm-head3-deltas-noatt.out
#
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -cuda -keypoints  hands > outs/bilstm-hands-deltas-noatt0.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -cuda -keypoints  hands > outs/bilstm-hands-deltas-noatt1.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -cuda -keypoints  hands > outs/bilstm-hands-deltas-noatt2.out
# CUDA_VISIBLE_DEVICES=1 python3 -u train_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -cuda -keypoints  hands > outs/bilstm-hands-deltas-noatt3.out

# run jointBiLSTM brute
cd ../JointBiLSTM
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 0 -input brute -keypoints full > outs/bilstm-full0-brute.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 1 -input brute -keypoints full > outs/bilstm-full1-brute.out
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 2 -input brute -keypoints full > outs/bilstm-full2-brute.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 3 -input brute -keypoints full > outs/bilstm-full3-brute.out

CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 0 -input brute -keypoints full-head > outs/bilstm-full--brutehead0.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 1 -input brute -keypoints full-head > outs/bilstm-full--brutehead1.out
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 2 -input brute -keypoints full-head > outs/bilstm-full--brutehead2.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 3 -input brute -keypoints full-head > outs/bilstm-full--brutehead3.out

CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 0 -input brute -keypoints full-hh > outs/bilstm-full--brutehh0.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 1 -input brute -keypoints full-hh > outs/bilstm-full--brutehh1.out
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 2 -input brute -keypoints full-hh > outs/bilstm-full--brutehh2.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 3 -input brute -keypoints full-hh > outs/bilstm-full--brutehh3.out

CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 0 -input brute -keypoints head > outs/bilstm-head0-brute.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 1 -input brute -keypoints head > outs/bilstm-head1-brute.out
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 2 -input brute -keypoints head > outs/bilstm-head2-brute.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 3 -input brute -keypoints head > outs/bilstm-head3-brute.out

CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 0 -input brute -keypoints hands > outs/bilstm-hands-brute0.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 1 -input brute -keypoints hands > outs/bilstm-hands-brute1.out
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 2 -input brute -keypoints hands > outs/bilstm-hands-brute2.out &
CUDA_VISIBLE_DEVICES=0 python3 -u train_joint_bilstm.py -epochs 20 -interp -cuda -emotion 3 -input brute -keypoints hands > outs/bilstm-hands-brute3.out


# run jointBiLSTM delta
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -keypoints full > outs/bilstm-full0-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -keypoints full > outs/bilstm-full1-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -keypoints full > outs/bilstm-full2-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -keypoints full > outs/bilstm-full3-deltas-noatt.out
#
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -keypoints full-hh > outs/bilstm-full--deltas-noatthh0.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -keypoints full-hh > outs/bilstm-full--deltas-noatthh1.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -keypoints full-hh > outs/bilstm-full--deltas-noatthh2.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -keypoints full-hh > outs/bilstm-full--deltas-noatthh3.out
#
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -keypoints full-head > outs/bilstm-full--deltas-noatthead0.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -keypoints full-head > outs/bilstm-full--deltas-noatthead1.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -keypoints full-head > outs/bilstm-full--deltas-noatthead2.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -keypoints full-head > outs/bilstm-full--deltas-noatthead3.out
#
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -keypoints head > outs/bilstm-head0-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -keypoints head > outs/bilstm-head1-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -keypoints head > outs/bilstm-head2-deltas-noatt.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -keypoints head > outs/bilstm-head3-deltas-noatt.out
#
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 0 -input deltas-noatt -keypoints hands > outs/bilstm-hands-deltas-noatt0.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 1 -input deltas-noatt -keypoints hands > outs/bilstm-hands-deltas-noatt1.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 2 -input deltas-noatt -keypoints hands > outs/bilstm-hands-deltas-noatt2.out
# CUDA_VISIBLE_DEVICES=6 python3 -u train_joint_bilstm.py -epochs 20 -interp -emotion 3 -input deltas-noatt -keypoints hands > outs/bilstm-hands-deltas-noatt3.out
