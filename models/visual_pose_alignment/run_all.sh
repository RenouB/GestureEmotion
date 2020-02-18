cd random
python3 classify_random.py
cd ../distance/voting
python3 classify_distance_voting.py
cd ../whole_image
python3 classify_distance_whole_image.py
cd ../../SVM/voting
python3 train_svm_voting.py
cd ../whole_image
python3 train_svm_whole_image.py
cd ../..
more random/outputs/scores/*csv | cat
more */*/*/scores/*csv | cat
