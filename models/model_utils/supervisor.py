import numpy as np
import argparse
import sys
import os
import torch
import re
import json
import time

from torch.nn.utils import clip_grad_norm

from ..data_utils import data_utils

CKPT_PATTERN = re.compile('^ckpt-(\d+)$')


class Supervisor(object):
	"""
	The base class to manage the high-level model execution processes. The concrete classes for different applications are derived from it.
	"""
	def __init__(self, model, args):
		self.data_processor = data_utils.DataProcessor(args)
		self.model = model
		self.keep_last_n = args.keep_last_n
		self.global_step = 0
		self.batch_size = args.batch_size
		self.model_dir = args.model_dir


	def load_pretrained(self, load_model):
		print("Read model parameters from %s." % load_model)
		checkpoint = torch.load(load_model)
		self.model.load_state_dict(checkpoint)


	def save_model(self):
		if not os.path.exists(self.model_dir):
			os.makedirs(self.model_dir)
		global_step_padded = format(self.global_step, '08d')
		ckpt_name = 'ckpt-' + global_step_padded
		path = os.path.join(self.model_dir, ckpt_name)
		ckpt = self.model.state_dict()
		torch.save(ckpt, path)

		if self.keep_last_n is not None:
			ckpts = []
			for file_name in os.listdir(self.model_dir):
				matched_name = CKPT_PATTERN.match(file_name)
				if matched_name is None or matched_name == ckpt_name:
					continue
				step = int(matched_name.group(1))
				ckpts.append((step, file_name))
			if len(ckpts) > self.keep_last_n:
				ckpts.sort()
				os.unlink(os.path.join(self.model_dir, ckpts[0][1]))


	def train(self, batch_input, batch_labels):
		self.model.optimizer.zero_grad()
		cur_loss, pred_logits, predictions = self.model(batch_input, batch_labels)
		gt_output = batch_input['gt']
		pred_acc = torch.sum(predictions == gt_output)
		pred_acc = pred_acc.item() * 1.0 / (gt_output.size()[0] * gt_output.size()[1])

		self.global_step += 1
		cur_loss.backward()
		self.model.train_step()
		return cur_loss.item(), pred_acc

	def eval(self, data, data_order_invariant=False, max_eval_size=None):
		self.model.eval()
		data_size = len(data)
		if max_eval_size is not None:
			data_size = min(data_size, max_eval_size)
		eval_data = data[:data_size]
		test_loss = 0.0
		test_label_acc = 0
		test_data_acc = 0
		test_acc = 0

		predictions = []
		for batch_idx in range(0, data_size, self.batch_size):
			batch_input, batch_labels = self.data_processor.get_batch(eval_data, self.batch_size, batch_idx)
			cur_loss, cur_pred_logits, cur_predictions = self.model(batch_input, batch_labels, eval_flag=True)
			test_loss += cur_loss.item() * batch_labels.size()[0]
			cur_predictions = cur_predictions.data.cpu().numpy().tolist()
			for i, sample in enumerate(batch_input['init_data']):
				gt_prog = self.data_processor.ids_to_prog(sample, sample['output_gt'])
				pred_prog = self.data_processor.ids_to_prog(sample, cur_predictions[i])
				gt_label = sample['label']
				pred_label = self.data_processor.label_extraction(pred_prog)
				if gt_label == pred_label:
					cur_test_label_acc = 1
				else:
					cur_test_label_acc = 0
				target_dfs, target_strs, target_vars = sample['target_dfs'], sample['target_strs'], sample['target_vars']
				pred_dfs, pred_strs, pred_vars, _ = self.data_processor.data_extraction(pred_prog,
					sample['reserved_dfs'], sample['reserved_strs'], sample['reserved_vars'])

				if data_order_invariant:
					if (set(target_dfs + target_strs + target_vars) == set(pred_dfs + pred_strs + pred_vars) and
						len(target_dfs + target_strs + target_vars) == len(pred_dfs + pred_strs + pred_vars)):
						cur_test_data_acc = 1
					else:
						cur_test_data_acc = 0
				else:
					if target_dfs + target_strs + target_vars == pred_dfs + pred_strs + pred_vars:
						cur_test_data_acc = 1
					else:
						cur_test_data_acc = 0
				cur_test_acc = min(cur_test_label_acc, cur_test_data_acc)
				test_label_acc += cur_test_label_acc
				test_data_acc += cur_test_data_acc
				test_acc += cur_test_acc
			print('batch_idx: ', batch_idx, 'test_label_acc: ', test_label_acc, 'test_data_acc', test_data_acc, 'test_acc', test_acc)
			predictions += cur_predictions

		test_loss /= data_size
		test_label_acc = test_label_acc * 1.0 / data_size
		test_data_acc = test_data_acc * 1.0 / data_size
		test_acc = test_acc * 1.0 / data_size
		self.model.train()
		return test_loss, test_label_acc, test_data_acc, test_acc, predictions
