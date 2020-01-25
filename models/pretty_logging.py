import os
import sys
import logging
import time

PROJECT_DIR = '/'.join(os.path.dirname(os.path.realpath(__file__)).split("/")[:-1])
sys.path.insert(0, PROJECT_DIR)
from definitions import constants
MODELS_DIR = constants["MODELS_DIR"]

def construct_basename(args):
	if args.joint:
		joint_str = 'JOINT'
	else:
		joint_str = 'IND'

	mode_str = str(args.modalities)

	lr_str = 'lr'+str(args.lr)
	l2_str = 'l2'+str(args.l2)
	dropout_str = 'dr'+str(args.dropout)
	epochs_str = 'ep'+str(args.epochs)
	basename = '-'.join([joint_str, mode_str, args.keypoints, lr_str, l2_str, dropout_str, epochs_str])

	if args.test:
		basename = 'TEST-'+basename
	return basename

def construct_crf_basename(args):
	if args.joint:
		joint_str = 'JOINT'
	else:
		joint_str = 'IND'

	mode_str = str(args.modalities)

	basename = '-'.join([joint_str, mode_str, args.keypoints])

	if args.test:
		basename = 'TEST-'+basename
	return basename

def get_write_dir(model_type, input_type, joint, modalities, emotion=None):
	if model_type == 'CNN':
		model_dir ='MultiChannelCNN'
	elif model_type == 'SVM':
		model_dir = 'SVM'
	elif model_type == 'random':
		model_dir = 'rand'

	if joint:
		joint_dir = 'joint'
	else:
		joint_dir ='ind'

	if modalities == 0:
		mode_dir = 'pose'
	elif modalities == 1:
		mode_dir = 'speech'
	else:
		mode_dir = 'both'

	if emotion == 0:
		emotion_str = 'anger'
	if emotion == 1:
		emotion_str = 'happiness'
	if emotion == 2:
		emotion_str = 'sadness'
	if emotion == 3:
		emotion_str = 'surprise'

	return os.path.join(MODELS_DIR, model_dir, input_type, emotion_str, joint_dir, mode_dir)

class PrettyLogger():
	def __init__(self, args, logs_dir, basename, starttime):
		self.starttime = starttime
		try:
			os.system("rm {}".format(os.path.join(logs_dir, basename+'.log')))
		except:
			pass
		logging.basicConfig(filename=os.path.join(logs_dir, basename+'.log'),
		 					level=logging.INFO, format='%(message)s')

		logging.info(str(args))
		# logging.info('\n')
		logging.info('{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'.format('',"epoch",
													"macro-p", "macro-r", "macro-f",
													"acc"))
		return

	def update_scores(self, scores, epoch, mode):
		scores_to_log = \
				[scores['macro_p'], scores['macro_r'], scores['macro_f'], scores['acc']]

		if mode != 'DEV':
			logging.info('{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'.format('',"epoch",
													"macro-p", "macro-r", "macro-f", "acc"))
		logging.info('{:>7}, {:>7}, {:>7.4f}, {:>7.4f}, {:>7.4f}, {:>7.4f}'.format(
						*[mode, epoch] + scores_to_log))

		if mode == 'DEV':
			logging.info('--------'*11)

		return

	def new_fold(self, k):
		logging.info('\n')
		logging.info(logging.info('FOLD '+str(k)+'--------'*10))

	def close(self, best_score, best_k):
		logging.info("\n")
		logging.info("STARTIME {}".format(self.starttime))
		logging.info("ENDTIME {}".format(time.strftime('%b-%d-%Y-%H%M')))
		logging.info("\n")
		logging.info("BEST_SCORE: {:.2f}".format(best_score))
		logging.info("AT K: {}".format(best_k))
		return
