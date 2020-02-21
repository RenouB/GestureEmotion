run run_joint_bilstm.sh

then scores_to_csv.py

then print_results.py

Output will have errors, but should have results for the JointBiLSTM model. These can be compared to expected_joint_output.txt.

Unfortunately, in expected_joint_output.txt, results for sadness are missing because of a last minute technical issue. You can still compare results of anger, happiness and surprise to your outputs.

For this reason expected log outputs have also been included. expected_sadness.log should be exactly the same as the sadness log output by your runs.

One thing to be careful about: in order to get expected outputs, you must be sure that the only .pkl files in emotion_classification directory are those output by the run_joint_bilstm.sh script.
