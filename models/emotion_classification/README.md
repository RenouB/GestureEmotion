run run_joint_bilstm.sh

then scores_to_csv.py

then print_results.py

Output will have errors, but should have results for the JointBiLSTM model. Compare this to expected_output.txt

Results will not be exactly the same as those in the report, because an issue with random seeds was fixed at the last minute.

One thing to be careful about: in order to get expected outputs, you must be sure that the only .pkl files in emotion_classification directory are those output by the run_joint_bilstm.sh script.
