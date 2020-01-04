import os
import sys
import numpy as np
import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

import argparse
import logging
from pretty_logging import PrettyLogger, construct_basename
import time




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-epochs', type=int, default=1)
	parser.add_argument('-batchsize', type=int, default=1)
	parser.add_argument('-lr', type=float, default=0.001)
	parser.add_argument('-l2', type=float, default=0.0001)
	parser.add_argument('-comment', default='')
	parser.add_argument('-cuda', default=False, action='store_true',  help='use cuda')
	parser.add_argument('-gpu', default=0, type=int, help='gpu id')
	parser.add_argument('-dropout', default=0.5, type=float, help='dropout probability')
	parser.add_argument('-debug', action='store_true', default=False)
	parser.add_argument('-test', action='store_true', default=False)
	parser.add_argument('-modalities', default=0)
	parser.add_argument('-joint', action=store_true, default=False)
	args = parser.parse_args()
	
	if args.debug:
		args.epochs = 1
		print_denominator = 10
	else:
		print_denominator = 100
	
	print('epochs', args.epochs)
	if args.cuda and torch.cuda.is_available():
		device = torch.device('cuda:'+str(args.gpu))
	else:
		device = torch.device('cpu')
	print("device: ", device)
	
	# basename for logs, weights
	startime = time.strftime('%b-%d-%Y-%H%M')
	basename = construct_basename(args, starttime)

	# initialize logging config
	logger = PrettyLogger(args, basename)

	if not args.joint:
		if args.modalities == 0:
			model =
		elif args.modalities == 1:
			model = 
		elif args.modalities == 2:
			model = 
	else:
		if args.modalities == 0:
			model = 
		elif args.modalities == 1:
			model = 
		elif args.modalities == 2:
			model = 


	if args.test:
		model.load(os.path.join(BERT_MODELS_DIR, 'weights', basename+'.weights'))
		model.eval()
		loss_fxn = torch.nn.CrossEntropyLoss()
		test_data = # DATASET

		test_epoch_loss = 0
		test_epoch_labels = []
		test_epoch_preds = []

		batch_counter = 0
		test_loader = DataLoader(test_data, shuffle=True, batch_size=args.batchsize)
		for dialog in test_loader:
			batch_outputs = []
			batch_labels = []
			if batch_counter % print_denominator == 0:
				print('Evaluating batch', batch_counter)
			# for dialog in batch:
			dialog_length = len(dialog['labels'])
			# start getting model
				out = model()
				batch_outputs.append(out)
				batch_labels.append(dialog['labels'][i])
				test_epoch_labels.append(dialog['labels'][i])
				test_epoch_preds.append(out.argmax().item())
			
			batch_outputs = torch.cat(batch_outputs, dim=0).to(device)
			batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
			batch_loss = loss_fxn(batch_outputs, batch_labels)
			test_epoch_loss += batch_loss.item()
			batch_counter += 1

		micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(test_epoch_labels, test_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='micro')
		macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(test_epoch_labels, test_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='macro')
		acc = sum(np.array(test_epoch_preds) == np.array(test_epoch_labels)) / len(test_epoch_labels)
		micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc = [100*round(item, 4) for item in \
																	   [micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc]]
		logging.info('{:7}, {:7}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}'.format('test  ', 1, micro_p, micro_r, micro_f1, macro_p,
																 macro_r, macro_f1, acc))
		sys.exit()

	loss_fxn = torch.nn.CrossEntropyLoss()
	optim = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)

	best_acc = -1
	for epoch in range(args.epochs):
		train_epoch_loss = 0
		train_epoch_labels = []
		train_epoch_preds = []

		if args.debug:
			train_data = BertDataset(slot=args.slot, tuned=args.tuned, embeddings=args.embeddings, mode='dev', debug=args.debug)
		else:
			train_data = BertDataset(slot=args.slot, tuned=args.tuned, embeddings=args.embeddings, mode='train')
		model.train()
		train_loader = DataLoader(train_data, shuffle=True, batch_size=args.batchsize)
		batch_counter = 0
		for dialog in train_loader:
			batch_outputs = []
			batch_labels = []
			if batch_counter % print_denominator == 0:
				print('Training batch', batch_counter)
			# for dialog in batch:
			dialog_length = len(dialog['labels'])
			for i in range(dialog_length):
				user_tokens = dialog['user_tokens'][i]
				system_tokens = dialog['system_tokens'][i]
				user_embs = dialog['user_embeddings'].squeeze(0)[i].to(device)
				system_embs = dialog['system_embeddings'].squeeze(0)[i].to(device)
				user_length = dialog['user_lengths'][i]
				system_length = dialog['system_lengths'][i]

				out = model(user_tokens, user_embs, user_length, system_tokens, system_embs, system_length, first_turn=i==0)
				batch_outputs.append(out)
				batch_labels.append(dialog['labels'][i])
				train_epoch_labels.append(dialog['labels'][i])
				train_epoch_preds.append(out.argmax().item())
			batch_outputs = torch.cat(batch_outputs, dim=0).to(device)
			batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
			batch_loss = loss_fxn(batch_outputs, batch_labels)
			train_epoch_loss += batch_loss.item()
			batch_counter += 1

			optim.zero_grad()
			batch_loss.backward()
			optim.step()

		micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(train_epoch_labels, train_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='micro')
		macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(train_epoch_labels, train_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='macro')
		acc = sum(np.array(train_epoch_preds) == np.array(train_epoch_labels)) / len(train_epoch_labels)
		micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc = [100*round(item, 4) for item in \
																	   [micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc]]
		logging.info('{:7}, {:7}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}'.format('train', epoch, micro_p, micro_r, micro_f1, macro_p,
																 macro_r, macro_f1, acc))

		writer.add_scalar('train/epoch_loss/', train_epoch_loss, epoch)
		writer.add_scalar('train/epoch_acc/', acc, epoch)


		print('##############################################')
		print('                Evaluating')
		print('##############################################')
		model.eval()

		dev_data = BertDataset(slot=args.slot, tuned=args.tuned, embeddings=args.embeddings, mode='dev', debug=args.debug)

		dev_epoch_loss = 0
		dev_epoch_labels = []
		dev_epoch_preds = []

		batch_counter = 0
		dev_loader = DataLoader(dev_data, shuffle=True, batch_size=args.batchsize)
		for dialog in dev_loader:
			batch_outputs = []
			batch_labels = []
			if batch_counter % print_denominator == 0:
				print('Evaluating batch', batch_counter)
			# for dialog in batch:
			dialog_length = len(dialog['labels'])
			for i in range(dialog_length):
				user_tokens = dialog['user_tokens'][i]
				system_tokens = dialog['system_tokens'][i]
				user_embs = dialog['user_embeddings'].squeeze(0)[i].to(device)
				system_embs = dialog['system_embeddings'].squeeze(0)[i].to(device)
				user_length = dialog['user_lengths'][i]
				system_length = dialog['system_lengths'][i]

				out = model(user_tokens, user_embs, user_length, system_tokens, system_embs, system_length, first_turn=i==0)
				batch_outputs.append(out)
				batch_labels.append(dialog['labels'][i])
				dev_epoch_labels.append(dialog['labels'][i])
				dev_epoch_preds.append(out.argmax().item())
			batch_outputs = torch.cat(batch_outputs, dim=0).to(device)
			batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(device)
			batch_loss = loss_fxn(batch_outputs, batch_labels)
			dev_epoch_loss += batch_loss.item()
			batch_counter += 1

		micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(dev_epoch_labels, dev_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='micro')
		macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(dev_epoch_labels, dev_epoch_preds, labels=[i for i in range(slot_value_counts[args.slot])], average='macro')
		acc = sum(np.array(dev_epoch_preds) == np.array(dev_epoch_labels)) / len(dev_epoch_labels)
		micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc = [100*round(item, 4) for item in \
																	   [micro_p, micro_r, micro_f1, macro_p, macro_r, macro_f1, acc]]
		logging.info('{:7}, {:7}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}, {:7.2f}'.format('dev  ', epoch, micro_p, micro_r, micro_f1, macro_p,
																 macro_r, macro_f1, acc))

		writer.add_scalar('dev/epoch_loss/', dev_epoch_loss, epoch)
		writer.add_scalar('dev/epoch_acc/', acc, epoch)

		if acc > best_acc:
			print('#########################################')
			print('New best accuracy: {} (previous {})'.format(acc, best_acc))
			print('saving model weights')
			print('#########################################')
			best_acc = acc
			torch.save(model.state_dict(), os.path.join(BERT_MODELS_DIR, 'weights', basename+'.weights'))


	end_timestamp = time.strftime('%b-%d-%Y_%H%M')
	logging.info('\nSTART {} END {}'.format(timestamp, end_timestamp))

