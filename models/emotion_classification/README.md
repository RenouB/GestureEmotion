run run_toy.sh

then scores_to_csv.py

then print_results.py

Output will have errors, but should have results for one JointBiLSTM model. Compare this to expected_toy_output.txt

If this works, run run_all.sh and repeat the same process. Compare to expected_joint_output.txt

Training time has been quite variable; anywhere from 7 - 16 hours.

The results obtained will be different than those presented in the report, because an issue with random seeds was only recently spotted and fixed.

The repo has been cloned and tested on a few different machines by a few different people, so it should run smoothly.

One issue that may arise: scores_to_csv.py will process all pickled scores files in the emotion_classification directory.
To get the exact same output as expected_joint_output.txt, you may need to delete any leftover pickle files from previous runs.

A nice command to do this:
find . -name \*.pkl -exec rm {} \;
