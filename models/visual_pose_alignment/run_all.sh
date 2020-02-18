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
echo "RANDOM"
cat random/outputs/scores/*csv
echo "DISTANCE WHOLE"
cat distance/whole_image/outputs/scores/*csv
echo "DISTANCE VOTE"
cat distance/voting/outputs/scores/*csv
echo "SVM WHOLE IMAGE"
cat SVM/whole_image/outputs/scores/*csv
echo "SVM VOTE"
cat SVM/voting/outputs/scores/*csv
