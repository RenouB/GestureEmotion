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

	if args.attention:
		att_str = 'ATT'
	else:
		att_str = 'NO-ATT'

	lr_str = 'lr'+str(args.lr)
	dropout_str = 'dr'+str(args.dropout)
	epochs_str = 'ep'+str(args.epochs)

	basename = '-'.join([joint_str, mode_str, att_str, lr_str, dropout_str, epochs_str])
	
	if args.test:
		basename = 'TEST-'+basename
	return basename

def get_logs_weights_scores_dirs(model_type, attention, joint, modalities):
	if model_type == 'CNN':
		model_dir ='MultiChannelCNN'
	if attention:
		attention_dir = 'attention'
	else:
		attention_dir = 'base'
	
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
	
	return os.path.join(MODELS_DIR, model_dir, 'outputs', 'logs', attention_dir,
				 joint_dir, mode_dir), \
		os.path.join(MODELS_DIR, model_dir, 'outputs', 'weights', attention_dir,
				 joint_dir, mode_dir), \
		os.path.join(MODELS_DIR, model_dir, 'outputs', 'scores', attention_dir,
				 joint_dir, mode_dir)

class PrettyLogger():
	def __init__(self, args, logs_dir, basename, starttime):
		self.starttime = starttime
		logging.basicConfig(filename=os.path.join(logs_dir, basename+'.log'),
		 					level=logging.INFO, format='%(message)s')

		logging.info(str(args))
		# logging.info('\n')

		return 

	def update_scores(self, scores, epoch, mode):
		scores_to_log = \
				[scores['macro_p'], scores['macro_r'], scores['macro_f'],
				scores['micro_p'], scores['micro_r'], scores['micro_f'], 
				scores['exact_acc']]
		
		if mode != 'DEV':
			logging.info('{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}'.format('',"epoch", 
													"macro-p", "macro-r", "macro-f",
													"micro-p", "micro-r","micro-f",  "acc"))
		logging.info('{:>7}, {:>7}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}'.format(
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