import logging
import time

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
	epochs_str = 'ep'+str(epochs)

	basename = '-'.join([joint_str, mode_str, att_str, lr_str, dropout_str, epochs_str])
	
	if args.test:
		basename = 'TEST-'+basename
	return basename

class PrettyLogger():
	def self.__init__(args, logs_dir, basename, startime):
		self.startime = startime
		logging.basicConfig(filename=os.path.join(logs_dir, 'logs',basename+'.log'),
		 					level=logging.INFO, format='%(message)s')

		logging.info(str(args))
		logging.info('\n')
		logging.info('{:7}, {:7}, {:7}, {:7}, {:7}, {:7}, {:7}, {:7}, {:7}'.format('',"epoch", 
													"micro-p", "micro-r","micro-f", 
													"macro-p", "macro-r", "macro-f", "acc"))

		return 

	def self.update_scores(mode, epoch, scores):
		micro_p, micro_r, micro_f, macro_p, macro_r, macro_f, acc = metrics
		logging.info('{:>7}, {:>7}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}, {:>7.2f}'.format(, 
													mode, epoch, micro_p, micro_r, micro_f,
													macro_p, macro_r, macro_f, acc))
		if mode == 'dev':
			logging.info('-------'*9)

		return

	def self.close(best_score):
		print("STARTIME {}".format(startime))
		print("ENDTIME {}".format(time.strftime('%b-%d-%Y-%H%M')))
		print("\n")
		print("BEST_SCORE: @{:.2f}".format(best_score))
		return