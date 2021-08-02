import numpy as np
import argparse
import sys
import os
import re
import json
import pandas as pd

class Logger(object):
	"""
	The class for recording the training process.
	"""
	def __init__(self, args):
		self.log_interval = args.log_interval
		self.log_dir = args.log_dir
		self.log_name = os.path.join(args.log_dir, args.log_name)
		self.best_eval_acc = 0
		self.records = []

	def write_summary(self, summary):
		print("global-step: %(global_step)d, train-acc: %(train_acc).3f, train-loss: %(train_loss).3f, eval-label-acc: %(eval_label_acc).3f, eval-data-acc: %(eval_data_acc).3f, eval-acc: %(eval_acc).3f, eval-loss: %(eval_loss).3f" % summary)
		self.records.append(summary)
		df = pd.DataFrame(self.records)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		df.to_csv(self.log_name, index=False)
		self.best_eval_acc = max(self.best_eval_acc, summary['eval_acc'])