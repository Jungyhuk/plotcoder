"""Data utils.
"""

import argparse
import collections
import json
import numpy as np
import os
import re
import string
import sys
import random
import enum
import six
import copy
from six.moves import map
from six.moves import range
from six.moves import zip
import ast
import ast2json

import torch
from torch.autograd import Variable

# Special vocabulary symbols
_PAD = b"_PAD"
_EOS = b"_EOS"
_GO = b"_GO"
_UNK = b"_UNK"
_DF = b"_DF"
_VAR = b"_VAR"
_STR = b"_STR"
_FUNC = b"_FUNC"
_VALUE = b"_VALUE"
_START_VOCAB = [_PAD, _EOS, _GO, _UNK, _DF, _VAR, _STR, _FUNC, _VALUE]

PAD_ID = 0
EOS_ID = 1
GO_ID = 2
UNK_ID = 3
DF_ID = 4
VAR_ID = 5
STR_ID = 6
FUNC_ID = 7
VALUE_ID = 8

def np_to_tensor(inp, output_type, cuda_flag):
	if output_type == 'float':
		inp_tensor = Variable(torch.FloatTensor(inp))
	elif output_type == 'int':
		inp_tensor = Variable(torch.LongTensor(inp))
	else:
		print('undefined tensor type')
	if cuda_flag:
		inp_tensor = inp_tensor.cuda()
	return inp_tensor

class DataProcessor(object):
	def __init__(self, args):
		self.word_vocab = json.load(open(args.word_vocab, 'r'))
		self.code_vocab = json.load(open(args.code_vocab, 'r'))
		self.word_vocab_list = _START_VOCAB[:]
		self.code_vocab_list = _START_VOCAB[:]
		self.vocab_offset = len(_START_VOCAB)
		for word in self.word_vocab:
			while self.word_vocab[word] + self.vocab_offset >= len(self.word_vocab_list):
				self.word_vocab_list.append(word)
			self.word_vocab_list[self.word_vocab[word] + self.vocab_offset] = word
		for word in self.code_vocab:
			while self.code_vocab[word] + self.vocab_offset >= len(self.code_vocab_list):
				self.code_vocab_list.append(word)
			self.code_vocab_list[self.code_vocab[word] + self.vocab_offset] = word
		self.word_vocab_size = len(self.word_vocab) + self.vocab_offset
		self.code_vocab_size = len(self.code_vocab) + self.vocab_offset
		self.cuda_flag = args.cuda
		self.nl = args.nl
		self.code_context = args.code_context
		self.use_comments = args.use_comments
		self.local_df_only = args.local_df_only
		self.target_code_transform = args.target_code_transform
		self.hierarchy = args.hierarchy
		self.copy_mechanism = args.copy_mechanism
		self.max_num_code_cells = args.max_num_code_cells
		self.max_word_len = args.max_word_len
		self.max_code_context_len = args.max_code_context_len
		self.max_decode_len = args.max_decode_len
		self.scatter_word_list = ['scatter', 'scatterplot', "'scatter'", '"scatter"', 'scatter_kws', "'o'", "'bo'", "'r+'", '"o"', '"bo"', '"r+"']
		self.hist_word_list = ['hist', "'hist'", '"hist"', 'bar', "'bar'", '"bar"', 'countplot', 'barplot']
		self.pie_word_list = ['pie', "'pie'", '"pie"']
		self.scatter_plot_word_list = ['lmplot', 'regplot']
		self.hist_plot_word_list = ['distplot', 'kdeplot', 'contour']
		self.normal_plot_word_list = ['plot']
		self.reserved_words = ['plt', 'sns']
		self.reserved_words += self.scatter_word_list + self.hist_word_list + self.pie_word_list + \
		self.scatter_plot_word_list + self.hist_plot_word_list + self.normal_plot_word_list
		for word in self.code_vocab:
			if self.code_vocab[word] < 1000 and word != 'subplot' and word[-4:] == 'plot' and not ('_' in word) and not (word in self.reserved_words):
				self.reserved_words.append(word)
		self.default_program_mask = [0] * len(self.code_vocab)
		for word in self.reserved_words:
			if word in self.code_vocab:
				self.default_program_mask[self.code_vocab[word]] = 1

	def label_extraction(self, code_seq):
		label = -1
		for word in self.scatter_word_list:
			if word in code_seq:
				label = 1
				break
		for word in self.hist_word_list:
			if word in code_seq:
				if label != -1:
					return -1
				label = 2
				break
		for word in self.pie_word_list:
			if word in code_seq:
				if label != -1:
					return -1
				label = 3
				break
		for word in self.scatter_plot_word_list:
			if word in code_seq:
				label = 4
				break
		for word in self.hist_plot_word_list:
			if word in code_seq:
				label = 5
				break
		for word in self.normal_plot_word_list:
			if word in code_seq:
				if 'scatter' in code_seq:
					label = 4
				elif 'hist' in code_seq or 'bar' in code_seq or 'countplot' in code_seq or 'barplot' in code_seq:
					label = 5
		if label == -1:
			label = 0
		return label

	def data_extraction(self, target_code_seq, reserved_dfs, reserved_strs, reserved_vars):
		target_dfs = []
		target_strs = []
		target_vars = []

		for i, tok in enumerate(target_code_seq):
			if i < len(target_code_seq) - 1 and target_code_seq[i + 1] == '=':
				continue
			if tok in reserved_dfs:
				target_dfs.append(tok)
			elif len(tok) > 2 and tok[0] in ["'", '"'] and tok[-1] in ["'", '"']:
				if tok[1:-1] in reserved_strs:
					target_strs.append(tok)
				elif i > 0 and i < len(target_code_seq) - 1 and target_code_seq[i - 1] == '[' and target_code_seq[i + 1] == ']':
					if i >= 3 and target_code_seq[i - 3] == '.':
						continue
					target_strs.append(tok)
				elif i >= 2 and target_code_seq[i - 1] == '=' and target_code_seq[i - 2] in ['x', 'y', 'data']:
					target_strs.append(tok)
			else:
				if tok in self.reserved_words:
					continue
				if tok in _START_VOCAB:
					continue
				if tok[0].isdigit() or tok[0] == '-' or '.' in tok:
					continue
				if tok in ['[', ']', '(', ')', '{', '}', 'ax', '=', '\n']:
					continue
				if i >= 2 and target_code_seq[i - 1] == '=' and target_code_seq[i - 2] not in ['x', 'y', 'data']:
					continue
				if i < len(target_code_seq) - 1 and target_code_seq[i + 1] == '[':
					if i > 0 and target_code_seq[i - 1] == '.':
						continue
					target_dfs.append(tok)
					reserved_dfs.append(tok)
					continue
				if i == len(target_code_seq) - 1 or target_code_seq[i + 1] not in ['.', ',', ')', ']']:
					continue
				if i < len(target_code_seq) - 2 and target_code_seq[i + 1] == '.' and target_code_seq[i + 2] in self.reserved_words and target_code_seq[i + 2] not in ['hist', 'pie']:
					continue
				target_vars.append(tok)
		return target_dfs, target_strs, target_vars, reserved_dfs


	def ids_to_prog(self, sample, ids):
		reserved_word_list = sample['reserved_dfs'] + sample['reserved_vars']
		for tok in sample['reserved_strs']:
			reserved_word_list.append("'" + tok + "'")
		prog = []
		for i in ids:
			if i < self.code_vocab_size:
				prog += [self.code_vocab_list[i]]
			else:
				prog += [reserved_word_list[i - self.code_vocab_size]]
			if i == EOS_ID:
				break
		return prog


	def get_joint_plot_type(self, init_label):
		if init_label in [0, 3]:
			return init_label
		elif init_label in [1, 4]:
			return 1
		else:
			return 2


	def load_data(self, filename):
		init_samples = json.load(open(filename, 'r'))
		samples = []
		for sample in init_samples:
			code_seq = sample['code_tokens']
			label = self.label_extraction(code_seq)
			samples.append(sample)
		return samples


	def ast_to_seq(self, ast):
		seq = []
		if ast['_type'] == 'Str':
			seq.append("'" + ast['s'] + "'")
			return seq
		if ast['_type'] == 'Name':
			seq.append(ast['id'])
			return seq
		if ast['_type'] == 'Attribute':
			if 'id' not in ast['value']:
				return seq
			df = ast['value']['id']
			attr = ast['attr']
			seq.append(df)
			seq.append('.')
			seq.append(attr)
			return seq
		if ast['_type'] == 'Subscript':
			if 'id' not in ast['value']:
				return seq
			df = ast['value']['id']
			if 'value' not in ast['slice']:
				seq.append(df)
				return seq
			attr = ast['slice']['value']
			attr = self.ast_to_seq(attr)
			if len(attr) == 0:
				return seq
			seq.append(df)
			seq.append('[')
			seq += attr
			seq.append(']')
			return seq
		return []

	def var_extraction(self, code, reserved_vars, reserved_dfs):
		cur_code = []
		for tok_idx, tok in enumerate(code):
			cur_code.append(tok)
			if tok == '\n':
				parse_error = False
				try:
					ast_tree = ast2json.ast2json(ast.parse(''.join(cur_code)))
				except:
					parse_error = True
				if parse_error:
					continue
				if len(ast_tree['body']) == 0:
					cur_code = []
					continue
				ast_tree = ast_tree['body'][0]
				if ast_tree['_type'] != 'Assign':
					cur_code = []
					continue
				var_list = ast_tree['targets']
				for var in var_list:
					if var['_type'] != 'Name':
						continue
					var_name = var['id']
					if var_name not in reserved_vars + reserved_dfs:
						reserved_vars.append(var_name)
				cur_code = []
		return reserved_vars

	def code_seq_transform(self, init_code_seq, reserved_dfs):
		code_seq = []
		st = 0

		while st < len(init_code_seq):
			ed = st
			cur_code_seq = []
			ast_tree = None
			while ed < len(init_code_seq) and init_code_seq[ed] != '\n':
				cur_code_seq.append(init_code_seq[ed])
				ed += 1
			while ed <= len(init_code_seq):
				if ed < len(init_code_seq):
					cur_code_seq.append(init_code_seq[ed])
					ed += 1
				parse_error = False
				try:
					ast_tree = ast2json.ast2json(ast.parse(''.join(cur_code_seq)))
				except:
					parse_error = True
				if not parse_error:
					break
				if ed == len(init_code_seq):
					break
			if ast_tree is None:
				st = ed
				continue
			if len(ast_tree['body']) == 0:
				st = ed
				continue
			ast_tree = ast_tree['body'][0]['value']
			if 'func' not in ast_tree:
				st = ed
				continue
			func = ast_tree['func']
			plot_type = self.label_extraction(cur_code_seq)
			if 'value' not in func or 'id' not in func['value']:
				st = ed
				continue
			func_name = func['value']['id']
			if func_name == 'sns':
				code_seq.append('sns')
			else:
				code_seq.append('plt')
			code_seq.append('.')
			if func['attr'] != 'plot':
				code_seq.append(func['attr'])
			else:
				if plot_type in [1, 4]:
					code_seq.append('scatter')
				elif plot_type in [2, 5]:
					code_seq.append('hist')
				elif plot_type == 3:
					code_seq.append('pie')
				else:
					code_seq.append('plot')
			code_seq.append('(')

			data_value = None
			if 'keywords' in ast_tree:
				kvs = ast_tree['keywords']
				for i in range(len(kvs)):
					kv = kvs[i]
					if kv['arg'] == 'data':
						data_value = kv['value']
						break

			if data_value is None:
				if func_name == 'sns' and len(ast_tree['args']) > 2:
					data_value = ast_tree['args'][2]
			
			if data_value is not None:
				data_value = self.ast_to_seq(data_value)
			else:
				if func_name in reserved_dfs:
					data_value = [func_name]

			x_value = None
			if 'keywords' in ast_tree:
				kvs = ast_tree['keywords']
				for i in range(len(kvs)):
					kv = kvs[i]
					if kv['arg'] == 'x':
						x_value = kv['value']
						break

			if x_value is None:
				if len(ast_tree['args']) > 0:
					x_value = ast_tree['args'][0]

			if x_value is not None:
				x_value = self.ast_to_seq(x_value)
				if len(x_value) == 0 or len(x_value) == 1 and x_value[0][0] in ["'", '"']:
					x_value = None
				else:
					if data_value is not None:
						code_seq += data_value
						code_seq.append('[')
					code_seq += x_value
					if data_value is not None:
						code_seq.append(']')

			y_value = None
			if 'keywords' in ast_tree:
				kvs = ast_tree['keywords']
				for i in range(len(kvs)):
					kv = kvs[i]
					if kv['arg'] == 'y':
						y_value = kv['value']
						break

			if y_value is None:
				if plot_type in [0, 1, 4] and len(ast_tree['args']) > 1:
					y_value = ast_tree['args'][1]

			if y_value is not None:
				y_value = self.ast_to_seq(y_value)
				if len(y_value) == 0 or len(y_value) == 1 and y_value[0][0] in ["'", '"']:
					y_value = None
				else:
					if code_seq[-1] not in ['(', ',']:
						code_seq.append(',')
					if data_value is not None:
						code_seq += data_value
						code_seq.append('[')
					code_seq += y_value
					if data_value is not None:
						code_seq.append(']')

			if code_seq[-1] == '(' or x_value is None or plot_type in [0, 1, 4] and y_value is None:
				while len(code_seq) > 0 and code_seq[-1] != '\n':
					code_seq = code_seq[:-1]
			if len(code_seq) > 0 and code_seq[-1] != '\n':
				code_seq.append(')')
				code_seq.append('\n')

			st = ed
		return code_seq

	def preprocess(self, samples):
		data = []
		indices = []
		cnt_word = 0
		cnt_code = 0
		max_target_code_seq_len = 0
		min_target_code_seq_len = 512
		for sample_idx, sample in enumerate(samples):
			init_code_seq = sample['code_tokens']
			api_seq = sample['api_sequence']

			code_seq = []
			for tok in init_code_seq:
				if len(tok) == 0 or tok[0] == '#':
					continue
				code_seq.append(tok)

			reserved_df_size = 0
			reserved_dfs = []
			reserved_df_attr_list = []
			reserved_str_size = 0
			reserved_strs = []
			reserved_vars = []
			reserved_var_size = 0

			code_context_cell_cnt = 0
			if self.local_df_only:
				max_num_code_cells = self.max_num_code_cells
			else:
				max_num_code_cells = len(sample['context'])
			for ctx_idx in range(len(sample['context'])):
				if code_context_cell_cnt == max_num_code_cells:
					break
				if not 'code_tokens' in sample['context'][ctx_idx]:
					continue
				cur_code_context = sample['context'][ctx_idx]['code_tokens']
				if type(cur_code_context) != list:
					continue
				code_context_cell_cnt += 1
				for i in range(len(cur_code_context)):
					if cur_code_context[i] in self.reserved_words:
						continue
					if i > 0 and i < len(cur_code_context) - 2 and cur_code_context[i] == '[' and cur_code_context[i + 1][0] in ["'", '"'] and cur_code_context[i + 2] == ']':
						if cur_code_context[i - 1] in ['[', ']', '(', ')', '=', ',']:
							continue
						if cur_code_context[i - 1] not in reserved_dfs:
							reserved_dfs.append(cur_code_context[i - 1])
							reserved_df_attr_list.append([])
					if i >= 4 and cur_code_context[i] == 'read_csv' and cur_code_context[i - 1] == '.' and cur_code_context[i - 2] == 'pd' and cur_code_context[i - 3] == '=':
						if cur_code_context[i - 4] in ['[', ']', '(', ')', '=', ',']:
							continue
						if not (cur_code_context[i - 4] in reserved_dfs):
							reserved_dfs.append(cur_code_context[i - 4])
							reserved_df_attr_list.append([])
					if i >= 4 and cur_code_context[i] == 'DataFrame' and cur_code_context[i - 1] == '.' and cur_code_context[i - 2] == 'pd' and cur_code_context[i - 3] == '=':
						if cur_code_context[i - 4] in ['[', ']', '(', ')', '=', ',']:
							continue
						if not (cur_code_context[i - 4] in reserved_dfs):
							reserved_dfs.append(cur_code_context[i - 4])
							reserved_df_attr_list.append([])
					if i >= 4 and cur_code_context[i] == 'DataReader' and cur_code_context[i - 1] == '.' and cur_code_context[i - 2] == 'data' and cur_code_context[i - 3] == '=':
						if cur_code_context[i - 4] in ['[', ']', '(', ')', '=', ',']:
							continue
						if not (cur_code_context[i - 4] in reserved_dfs):
							reserved_dfs.append(cur_code_context[i - 4])
							reserved_df_attr_list.append([])
					if i >= 2 and i < len(cur_code_context) - 2 and cur_code_context[i] == 'head' and cur_code_context[i - 1] == '.' \
					and cur_code_context[i + 1] == '(' and cur_code_context[i + 2] == ')':
						if cur_code_context[i - 2] in ['[', ']', '(', ')', '=', ',']:
							continue
						if not (cur_code_context[i - 2] in reserved_dfs):
							reserved_dfs.append(cur_code_context[i - 2])
							reserved_df_attr_list.append([])
					if i >= 4 and cur_code_context[i] == 'load' and cur_code_context[i - 1] == '.' and cur_code_context[i - 2] == 'np' and cur_code_context[i - 3] == '=':
						if cur_code_context[i - 4] in ['[', ']', '(', ')', '=', ',']:
							continue
						if not (cur_code_context[i - 4] in reserved_dfs):
							reserved_dfs.append(cur_code_context[i - 4])
							reserved_df_attr_list.append([])
			
			code_context = []
			code_context_cell_cnt = 0
			for ctx_idx in range(len(sample['context'])):
				if code_context_cell_cnt == max_num_code_cells:
					break
				if not 'code_tokens' in sample['context'][ctx_idx]:
					continue
				init_cur_code_context = sample['context'][ctx_idx]['code_tokens']
				if type(init_cur_code_context) != list:
					continue
				cur_code_context = []
				for tok in init_cur_code_context:
					if len(tok) == 0 or tok[0] == '#':
						continue
					cur_code_context.append(tok)
				selected_code_context = []
				i = 0
				while i < len(cur_code_context):
					if cur_code_context[i] in reserved_dfs:
						df_idx = reserved_dfs.index(cur_code_context[i])
						selected = False
						csv_reading = False
						st = i - 1
						while st >= 0 and cur_code_context[st] != '\n':
							if cur_code_context[st] == 'read_csv':
								csv_reading = True
							st -= 1
						ed = i + 1
						while ed < len(cur_code_context) and cur_code_context[ed] != '\n':
							if cur_code_context[ed] == 'read_csv':
								csv_reading = True
							ed += 1
						while ed < len(cur_code_context):
							if cur_code_context[ed] == 'read_csv':
								csv_reading = True

							parse_error = False
							try:
								ast_tree = ast2json.ast2json(ast.parse(''.join(cur_code_context[st + 1:ed + 1])))
							except:
								parse_error = True
							if not parse_error:
								break
							ed += 1
						if csv_reading:
							i = ed + 1
							continue
						for tok_idx in range(st + 1, ed):
							if cur_code_context[tok_idx] in self.reserved_words:
								continue
							if len(cur_code_context[tok_idx]) > 2 and cur_code_context[tok_idx][0] in ["'", '"'] and cur_code_context[tok_idx][-1] in ["'", '"'] and not (cur_code_context[tok_idx][1:-1] in reserved_df_attr_list[df_idx]):
								if not ('.csv' in cur_code_context[tok_idx] or cur_code_context[tok_idx - 1] == '='):
									reserved_df_attr_list[df_idx].append(cur_code_context[tok_idx][1:-1])
									if cur_code_context[tok_idx][1:-1] not in reserved_strs:
										reserved_strs.append(cur_code_context[tok_idx][1:-1])
										reserved_str_size += 1
								selected = True
						if selected:
							if len(selected_code_context) > 0 and selected_code_context[-1] != '\n':
								selected_code_context += ['\n']
							selected_code_context = selected_code_context + cur_code_context[st + 1: ed + 1]
							i = ed + 1
							continue

					if cur_code_context[i] == 'savez':
						st = i - 1
						while st >= 0 and cur_code_context[st] != '\n':
							st -= 1
						ed = i + 1
						while ed < len(cur_code_context) and cur_code_context[ed] != '\n':
							ed += 1
						while ed < len(cur_code_context):
							parse_error = False
							try:
								ast_tree = ast2json.ast2json(ast.parse(''.join(cur_code_context[st + 1:ed + 1])))
							except:
								parse_error = True
							if not parse_error:
								break
							ed += 1

						if 'body' not in ast_tree:
							i += 1
							continue

						ast_tree = ast_tree['body'][0]['value']
						if 'keywords' in ast_tree:
							kvs = ast_tree['keywords']
							for i in range(len(kvs)):
								kv = kvs[i]
								if kv['arg'] not in reserved_strs:
									reserved_strs.append(kv['arg'])
									reserved_str_size += 1
							selected = True

						if selected:
							if len(selected_code_context) > 0 and selected_code_context[-1] != '\n':
								selected_code_context += ['\n']
							selected_code_context = selected_code_context + cur_code_context[st + 1: ed + 1]
							i = ed + 1
							continue
					i += 1
				if code_context_cell_cnt < self.max_num_code_cells:
					if len(code_context) > 0 and len(cur_code_context) > 0 and cur_code_context[-1] != '\n':
						code_context = ['\n'] + code_context
					reserved_vars = self.var_extraction(cur_code_context, reserved_vars, reserved_dfs)
					code_context = cur_code_context + code_context
					code_context_cell_cnt += 1
				else:
					if len(code_context) > 0 and len(selected_code_context) > 0 and selected_code_context[-1] != '\n':
						code_context = ['\n'] + code_context
					code_context = selected_code_context + code_context
					reserved_vars = self.var_extraction(selected_code_context, reserved_vars, reserved_dfs)

			keyword_pos = code_seq.index('plt')
			while code_seq[keyword_pos + 1] != '.':
				keyword_pos = keyword_pos + 1 + code_seq[keyword_pos + 1:].index('plt')
			if 'sns' in code_seq:
				keyword_pos = min(keyword_pos, code_seq.index('sns'))
			if len(code_context) > 0 and code_context[-1] != '\n':
				code_context += ['\n']
			code_context += code_seq[:keyword_pos]

			i = 0
			while i < keyword_pos:
				if code_seq[i] in reserved_dfs:
					df_idx = reserved_dfs.index(code_seq[i])
					csv_reading = False
					st = i - 1
					while st >= 0 and code_seq[st] != '\n':
						if code_seq[st] == 'read_csv':
							csv_reading = True
						st -= 1
					ed = i + 1
					while ed < keyword_pos and code_seq[ed] != '\n':
						if code_seq[ed] == 'read_csv':
							csv_reading = True
						ed += 1
					while ed < keyword_pos:
						if code_seq[ed] == 'read_csv':
							csv_reading = True

						parse_error = False
						try:
							ast_tree = ast2json.ast2json(ast.parse(''.join(code_seq[st + 1:ed + 1])))
						except:
							parse_error = True
						if not parse_error:
							break
						ed += 1
					if csv_reading:
						i = ed + 1
						continue
					for tok_idx in range(st + 1, ed):
						if code_seq[tok_idx] in self.reserved_words:
							continue
						if len(code_seq[tok_idx]) > 2 and code_seq[tok_idx][0] in ["'", '"'] and code_seq[tok_idx][-1] in ["'", '"'] and not (code_seq[tok_idx][1:-1] in reserved_df_attr_list[df_idx]):
							if not ('.csv' in code_seq[tok_idx] or code_seq[tok_idx - 1] == '='):
								reserved_df_attr_list[df_idx].append(code_seq[tok_idx][1:-1])
								if code_seq[tok_idx][1:-1] not in reserved_strs:
									reserved_strs.append(code_seq[tok_idx][1:-1])
									reserved_str_size += 1
					i = ed + 1
					continue
							
				if code_seq[i] == 'savez':
					st = i - 1
					while st >= 0 and code_seq[st] != '\n':
						st -= 1
					ed = i + 1
					while ed < keyword_pos and code_seq[ed] != '\n':
						ed += 1
					while ed < keyword_pos:
						parse_error = False
						try:
							ast_tree = ast2json.ast2json(ast.parse(''.join(code_seq[st + 1:ed + 1])))
						except:
							parse_error = True
						if not parse_error:
							break
						ed += 1

					if 'body' not in ast_tree:
						i += 1
						continue

					ast_tree = ast_tree['body'][0]['value']

					if 'keywords' in ast_tree:
						kvs = ast_tree['keywords']
						for i in range(len(kvs)):
							kv = kvs[i]
							if kv['arg'] not in reserved_strs:
								reserved_strs.append(kv['arg'])
								reserved_str_size += 1
					i = ed + 1
					continue
				i += 1

			code_seq = code_seq[keyword_pos:]
			target_code_seq = []
			selected_code_idx = 0
			code_idx = 0
			while code_idx < len(code_seq):
				tok = code_seq[code_idx]
				if not (tok in self.reserved_words):
					code_idx += 1
					continue
				if code_idx == len(code_seq) - 1 or code_seq[code_idx + 1] != '(':
					code_idx += 1
					continue
				st_idx = code_idx - 1
				while st_idx >= 0 and code_seq[st_idx] != '\n':
					st_idx -= 1
				ed_idx = code_idx + 2
				include_function_calls = False
				while ed_idx < len(code_seq) and code_seq[ed_idx] != ')':
					if code_seq[ed_idx] == '(':
						include_function_calls = True
						break
					ed_idx += 1
				if include_function_calls:
					code_idx += 1
					continue
				while ed_idx < len(code_seq) and code_seq[ed_idx] != '\n':
					ed_idx += 1
				target_code_seq += code_seq[st_idx + 1: ed_idx + 1]
				code_context += code_seq[selected_code_idx:st_idx + 1]

				i = selected_code_idx
				while i <= st_idx:
					if code_seq[i] in reserved_dfs:
						df_idx = reserved_dfs.index(code_seq[i])
						csv_reading = False
						st = i - 1
						while st >= selected_code_idx and code_seq[st] != '\n':
							if code_seq[st] == 'read_csv':
								csv_reading = True
							st -= 1
						ed = i + 1
						while ed <= st_idx and code_seq[ed] != '\n':
							if code_seq[ed] == 'read_csv':
								csv_reading = True
							ed += 1
						while ed <= st_idx:
							if code_seq[ed] == 'read_csv':
								csv_reading = True

							parse_error = False
							try:
								ast_tree = ast2json.ast2json(ast.parse(''.join(code_seq[st + 1:ed + 1])))
							except:
								parse_error = True
							if not parse_error:
								break
							ed += 1
						if csv_reading:
							i = ed + 1
							continue
						for tok_idx in range(st + 1, ed):
							if code_seq[tok_idx] in self.reserved_words:
								continue
							if len(code_seq[tok_idx]) > 2 and code_seq[tok_idx][0] in ["'", '"'] and code_seq[tok_idx][-1] in ["'", '"'] and not (code_seq[tok_idx][1:-1] in reserved_df_attr_list[df_idx]):
								if not ('.csv' in code_seq[tok_idx] or code_seq[tok_idx - 1] == '='):
									reserved_df_attr_list[df_idx].append(code_seq[tok_idx][1:-1])
									if code_seq[tok_idx][1:-1] not in reserved_strs:
										reserved_strs.append(code_seq[tok_idx][1:-1])
										reserved_str_size += 1
						i = ed + 1
						continue
					if code_seq[i] == 'savez':
						st = i - 1
						while st >= selected_code_idx and code_seq[st] != '\n':
							st -= 1
						ed = i + 1
						while ed <= st_idx and code_seq[ed] != '\n':
							ed += 1
						while ed <= st_idx:
							parse_error = False
							try:
								ast_tree = ast2json.ast2json(ast.parse(''.join(code_seq[st + 1:ed + 1])))
							except:
								parse_error = True
							if not parse_error:
								break
							ed += 1

						if 'body' not in ast_tree:
							i += 1
							continue
						ast_tree = ast_tree['body'][0]['value']
						if 'keywords' in ast_tree:
							kvs = ast_tree['keywords']
							for i in range(len(kvs)):
								kv = kvs[i]
								if kv['arg'] not in reserved_strs:
									reserved_strs.append(kv['arg'])
									reserved_str_size += 1
						i = ed + 1
						continue
					i += 1

				selected_code_idx = ed_idx + 1
				code_idx = ed_idx + 1
			
			init_target_code_seq = target_code_seq[:]
			target_code_seq = self.code_seq_transform(target_code_seq, reserved_dfs)

			label = self.label_extraction(target_code_seq)
			if label == -1:
				continue
			
			if len(target_code_seq) <= 5:
				continue

			if not self.target_code_transform:
				target_code_seq = init_target_code_seq[:]

			max_target_code_seq_len = max(max_target_code_seq_len, len(target_code_seq))
			min_target_code_seq_len = min(min_target_code_seq_len, len(target_code_seq))

			input_word_seq = []

			if self.nl and not self.use_comments:
				nl = sample['nl']
				nl = nl[:self.max_word_len - 1]
			elif self.use_comments and not self.nl:
				nl = sample['comments']
				nl = nl[:self.max_word_len - 1]
			elif not self.nl and not self.use_comments:
				nl = []
			else:
				nl = sample['nl'] + sample['comments']
				if len(nl) > self.max_word_len - 1:
					if len(sample['comments']) <= self.max_word_len // 2:
						nl = sample['nl'][:self.max_word_len - 1 - len(sample['comments'])] + sample['comments']
					elif len(sample['nl']) <= self.max_word_len // 2:
						nl = sample['nl'] + sample['comments'][:self.max_word_len - 1 - len(sample['nl'])]
					else:
						nl = sample['nl'][:self.max_word_len // 2 - 1] + sample['comments'][:self.max_word_len // 2]

			if not self.code_context:
				code_context = []
			if len(code_context) > self.max_code_context_len - 1:
				code_context = code_context[1 - self.max_code_context_len:]

			target_dfs, target_strs, target_vars, reserved_dfs = self.data_extraction(target_code_seq, reserved_dfs, reserved_strs, reserved_vars)

			init_reserved_dfs = list(reserved_dfs)
			for tok in init_reserved_dfs:
				if not (tok in code_context):
					reserved_dfs.remove(tok)
			reserved_df_size = len(reserved_dfs)

			init_reserved_strs = list(reserved_strs)
			for tok in init_reserved_strs:
				if not ('"' + tok + '"' in code_context or "'" + tok + "'" in code_context or tok in code_context):
					reserved_strs.remove(tok)
			reserved_str_size = len(reserved_strs)

			init_reserved_vars = list(reserved_vars)
			for tok in init_reserved_vars:
				if not (tok in code_context):
					reserved_vars.remove(tok)
			reserved_var_size = len(reserved_vars)

			input_code_seq = []
			input_code_nl_indices = []
			input_code_df_seq = []
			input_code_var_seq = []
			input_code_str_seq = []

			for i in range(len(code_context)):
				tok = code_context[i]
				input_code_nl_indices.append([])

				if tok in _START_VOCAB:
					input_code_seq.append(_START_VOCAB.index(tok))
					input_code_df_seq.append(-1)
					input_code_var_seq.append(-1)
					input_code_str_seq.append(-1)
					continue

				nl_lower = [tok.lower() for tok in nl]
				if tok.lower() in nl_lower:
					input_code_nl_indices[-1].append(nl_lower.index(tok.lower()))
				elif ("'" + tok.lower() + "'") in nl_lower:
					input_code_nl_indices[-1].append(nl_lower.index("'" + tok.lower() + "'"))
				elif ('"' + tok.lower() + '"') in nl_lower:
					input_code_nl_indices[-1].append(nl_lower.index('"' + tok.lower() + '"'))
				elif tok[0] in ["'", '"'] and tok[-1] in ["'", '"'] and tok[1:-1].lower() in nl_lower:
					input_code_nl_indices[-1].append(nl_lower.index(tok[1:-1].lower()))
				elif '_' in tok.lower():
					if tok[0] in ["'", '"'] and tok[-1] in ["'", '"']:
						tok_list = tok[1:-1].split('_')
					else:
						tok_list = tok.split('_')
					for sub_tok in tok_list:
						if sub_tok.lower() in nl_lower:
							input_code_nl_indices[-1].append(nl_lower.index(sub_tok.lower()))
				if len(input_code_nl_indices[-1]) > 2:
					input_code_nl_indices[-1] = input_code_nl_indices[-1][:2]
				elif len(input_code_nl_indices[-1]) < 2:
					input_code_nl_indices[-1] = input_code_nl_indices[-1] + [len(nl)] * (2 - len(input_code_nl_indices[-1]))

				if tok in self.code_vocab:
					input_code_seq.append(self.code_vocab[tok] + self.vocab_offset)

					if tok in reserved_dfs:
						input_code_df_seq.append(reserved_dfs.index(tok))
					else:
						input_code_df_seq.append(-1)
					if tok in reserved_vars:
						input_code_var_seq.append(reserved_vars.index(tok))
					else:
						input_code_var_seq.append(-1)
					if len(tok) > 2 and tok[0] in ["'", '"'] and tok[-1] in ["'", '"'] and tok[1:-1] in reserved_strs:
						input_code_str_seq.append(reserved_strs.index(tok[1:-1]))
					else:
						input_code_str_seq.append(-1)
				elif tok in reserved_dfs:
					input_code_seq.append(DF_ID)
					input_code_df_seq.append(reserved_dfs.index(tok))
					input_code_var_seq.append(-1)
					input_code_str_seq.append(-1)
				elif tok in reserved_vars:
					input_code_seq.append(VAR_ID)
					input_code_df_seq.append(-1)
					input_code_var_seq.append(reserved_vars.index(tok))
					input_code_str_seq.append(-1)
				elif tok[-1] in ["'", '"']:
					input_code_seq.append(STR_ID)
					if len(tok) > 2 and tok[0] in ["'", '"'] and tok[-1] in ["'", '"'] and tok[1:-1] in reserved_strs:
						input_code_str_seq.append(reserved_strs.index(tok[1:-1]))
					else:
						input_code_str_seq.append(-1)
					input_code_df_seq.append(-1)
					input_code_var_seq.append(-1)
				elif tok[0].isdigit() or tok[0] == '-' or '.' in tok:
					input_code_seq.append(VALUE_ID)
					input_code_df_seq.append(-1)
					input_code_var_seq.append(-1)
					input_code_str_seq.append(-1)
				elif i < len(code_context) - 1 and code_context[i + 1] == '(':
					input_code_seq.append(FUNC_ID)
					input_code_df_seq.append(-1)
					input_code_var_seq.append(-1)
					input_code_str_seq.append(-1)
				else:
					reserved_vars.append(tok)
					reserved_var_size += 1
					input_code_seq.append(VAR_ID)
					input_code_df_seq.append(-1)
					input_code_var_seq.append(reserved_vars.index(tok))
					input_code_str_seq.append(-1)

			if not self.copy_mechanism:
				for i in range(len(input_code_seq)):
					if input_code_seq[i] == DF_ID and input_code_df_seq[i] != -1:
						input_code_seq[i] = self.code_vocab_size + input_code_df_seq[i]
					elif input_code_seq[i] == VAR_ID and input_code_var_seq[i] != -1:
						input_code_seq[i] = self.code_vocab_size + reserved_df_size + input_code_var_seq[i]
					elif input_code_seq[i] == STR_ID and input_code_str_seq[i] != -1:
						input_code_seq[i] = self.code_vocab_size + reserved_df_size + reserved_var_size + input_code_str_seq[i]

			for word in nl:
				if word in self.word_vocab:
					input_word_seq.append(self.word_vocab[word] + self.vocab_offset)
				elif word in _START_VOCAB:
					input_word_seq.append(_START_VOCAB.index(word))
				elif word in reserved_vars:
					input_word_seq.append(VAR_ID)
				elif word in reserved_dfs:
					input_word_seq.append(DF_ID)
				elif word in reserved_strs:
					str_idx = reserved_strs.index(word)
					input_word_seq.append(STR_ID)
				elif word[0] in ["'", '"'] and word[-1] in ["'", '"']:
					tok = word[1:-1]
					if tok in reserved_vars:
						input_word_seq.append(VAR_ID)
					elif tok in reserved_dfs:
						input_word_seq.append(DF_ID)
					else:
						input_word_seq.append(UNK_ID)
				else:
					input_word_seq.append(UNK_ID)

			output_code_seq = []
			output_code_df_seq = []
			output_code_var_seq = []
			output_code_str_seq = []
			output_gt = []

			for i, tok in enumerate(target_code_seq):
				if tok in self.code_vocab:
					output_code_seq.append(self.code_vocab[tok] + self.vocab_offset)
					if tok in reserved_dfs:
						if self.hierarchy:
							output_code_seq[-1] = DF_ID
						output_code_df_seq.append(self.code_vocab_size + reserved_dfs.index(tok))
					else:
						output_code_df_seq.append(-1)
					
					if tok in target_vars and tok in code_context:
						if self.hierarchy:
							output_code_seq[-1] = VAR_ID
						output_code_var_seq.append(self.code_vocab[tok] + self.vocab_offset)
					else:
						output_code_var_seq.append(-1)

					if len(tok) > 2 and tok[0] in ["'", '"'] and tok[-1] in ["'", '"'] and tok[1:-1] in reserved_strs:
						if self.hierarchy:
							output_code_seq[-1] = STR_ID
						output_code_str_seq.append(self.code_vocab_size + reserved_df_size + reserved_var_size + reserved_strs.index(tok[1:-1]))
					elif tok in code_context and not (tok in reserved_dfs + reserved_vars + reserved_strs + self.reserved_words + sample['imports']) and tok[-1] in ['"', '"']:
						output_code_str_seq.append(self.code_vocab[tok] + self.vocab_offset)
					else:
						output_code_str_seq.append(-1)
				elif tok in _START_VOCAB:
					output_code_seq.append(_START_VOCAB.index(tok))
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				elif tok in reserved_dfs:
					df_idx = reserved_dfs.index(tok)
					if self.hierarchy:
						output_code_seq.append(DF_ID)
					else:
						output_code_seq.append(self.code_vocab_size + df_idx)
					output_code_df_seq.append(self.code_vocab_size + df_idx)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				elif tok in reserved_vars:
					var_idx = reserved_vars.index(tok)
					if self.hierarchy:
						output_code_seq.append(VAR_ID)
					else:
						output_code_seq.append(self.code_vocab_size + reserved_df_size + var_idx)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(self.code_vocab_size + reserved_df_size + var_idx)
					output_code_str_seq.append(-1)
				elif len(tok) > 2 and tok[0] in ["'", '"'] and tok[-1] in ["'", '"'] and tok[1:-1] in reserved_strs:
					str_idx = reserved_strs.index(tok[1:-1])
					if self.hierarchy:
						output_code_seq.append(STR_ID)
					else:
						output_code_seq.append(self.code_vocab_size + reserved_df_size + reserved_var_size + str_idx)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(self.code_vocab_size + reserved_df_size + reserved_var_size + str_idx)
				elif tok[-1] in ["'", '"']:
					output_code_seq.append(PAD_ID)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				elif tok[0].isdigit() or tok[0] == '-' or '.' in tok:
					output_code_seq.append(PAD_ID)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				elif i < len(target_code_seq) - 1 and target_code_seq[i + 1] in ['(', '=']:
					output_code_seq.append(PAD_ID)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				else:
					output_code_seq.append(PAD_ID)
					output_code_df_seq.append(-1)
					output_code_var_seq.append(-1)
					output_code_str_seq.append(-1)
				if output_code_seq[-1] == DF_ID:
					output_gt.append(output_code_df_seq[-1])
				elif output_code_seq[-1] == VAR_ID:
					output_gt.append(output_code_var_seq[-1])
				elif output_code_seq[-1] == STR_ID:
					output_gt.append(output_code_str_seq[-1])
				else:
					output_gt.append(output_code_seq[-1])

			input_word_seq += [EOS_ID]
			input_code_seq += [EOS_ID]
			output_code_seq += [EOS_ID]
			output_gt += [EOS_ID]
			output_code_df_seq += [-1]
			output_code_var_seq += [-1]
			output_code_str_seq += [-1]
			output_code_mask = [1] * 3 + [0] * (self.vocab_offset - 3)
			output_code_mask += list(self.default_program_mask)

			for tok in input_code_seq:
				if tok < self.code_vocab_size:
					output_code_mask[tok] = 1

			for tok in output_code_seq:
				if tok < self.code_vocab_size:
					output_code_mask[tok] = 1

			if not self.hierarchy:
				output_code_mask += [1] * (reserved_df_size + reserved_var_size + reserved_str_size)
			else:
				output_code_mask += [0] * (reserved_df_size + reserved_var_size + reserved_str_size)

			output_df_mask = [0] * (self.code_vocab_size + reserved_df_size + reserved_var_size + reserved_str_size)
			for df_idx in range(reserved_df_size):
				output_df_mask[self.code_vocab_size + df_idx] = 1
				if reserved_dfs[df_idx] in self.code_vocab:
					output_df_mask[self.code_vocab[reserved_dfs[df_idx]] + self.vocab_offset] = 1

			output_var_mask = [0] * (self.code_vocab_size + reserved_df_size + reserved_var_size + reserved_str_size)
			for var_idx in range(reserved_var_size):
				output_var_mask[self.code_vocab_size + reserved_df_size + var_idx] = 1
			for tok in code_context:
				if tok in self.code_vocab and tok in target_vars:
					output_var_mask[self.code_vocab[tok] + self.vocab_offset] = 1
				if tok in self.code_vocab and not (tok in reserved_dfs + reserved_vars + reserved_strs + self.reserved_words + sample['imports']) and not (tok[-1] in ['"', '"']) and not (tok[0].isdigit() or tok[0] == '-' or '.' in tok) and (i == len(target_code_seq) - 1 or target_code_seq[i + 1] != '('):
					output_var_mask[self.code_vocab[tok] + self.vocab_offset] = 1

			output_str_mask = [0] * (self.code_vocab_size + reserved_df_size + reserved_var_size + reserved_str_size)
			for str_idx in range(reserved_str_size):
				output_str_mask[self.code_vocab_size + reserved_df_size + reserved_var_size + str_idx] = 1
			for tok in code_context:
				if tok in self.code_vocab and not (tok in reserved_dfs + reserved_vars + reserved_strs + self.reserved_words + sample['imports']) and tok[-1] in ['"', '"']:
					output_str_mask[self.code_vocab[tok] + self.vocab_offset] = 1

			for i in range(3, self.vocab_offset):
				output_df_mask[i] = 0
				output_var_mask[i] = 0
				output_str_mask[i] = 0
				if not self.hierarchy:
					output_code_mask[i] = 0

			output_code_indices = []
			output_code_ctx_indices = []
			for tok in self.code_vocab_list:
				if tok in code_context:
					output_code_ctx_indices.append(code_context.index(tok))
				else:
					output_code_ctx_indices.append(len(code_context))

			output_code_nl_indices = []
			for tok in self.code_vocab_list:
				output_code_nl_indices.append([])
				if tok in _START_VOCAB:
					output_code_nl_indices[-1] += [len(nl), len(nl)]
				elif tok in self.scatter_word_list:
					nl_lower = [tok.lower() for tok in nl]
					if 'scatter' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('scatter'), len(nl)]
					elif 'scatterplot' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('scatterplot'), len(nl)]
					else:
						output_code_nl_indices[-1] += [len(nl), len(nl)]
				elif tok in self.hist_word_list:
					nl_lower = [tok.lower() for tok in nl]
					if 'histogram' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('histogram'), len(nl)]
					elif 'histograms' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('histograms'), len(nl)]
					else:
						output_code_nl_indices[-1] += [len(nl), len(nl)]
				elif tok in self.pie_word_list:
					nl_lower = [tok.lower() for tok in nl]
					if 'pie' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('pie'), len(nl)]
					else:
						output_code_nl_indices[-1] += [len(nl), len(nl)]
				elif tok in self.scatter_plot_word_list:
					nl_lower = [tok.lower() for tok in nl]
					if 'scatter' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('scatter')]
					if 'line' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('line')]
					elif 'linear' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('linear')]
					if len(output_code_nl_indices[-1]) < 2:
						output_code_nl_indices[-1] += [len(nl)] * (2 - len(output_code_nl_indices[-1]))
				elif tok in self.hist_plot_word_list:
					nl_lower = [tok.lower() for tok in nl]
					if 'distribution' in nl_lower:
						output_code_nl_indices[-1] += [nl_lower.index('distribution'), len(nl)]
					else:
						output_code_nl_indices[-1] += [len(nl), len(nl)]
				elif tok in code_context:
					output_code_nl_indices[-1] += input_code_nl_indices[code_context.index(tok)]
				else:
					output_code_nl_indices[-1] += [len(nl), len(nl)]


			for tok in reserved_dfs + reserved_vars:
				output_code_indices.append(code_context.index(tok))
			for tok in reserved_strs:
				if "'" + tok + "'" in code_context:
					output_code_indices.append(code_context.index("'" + tok + "'"))
				elif '"' + tok + '"' in code_context:
					output_code_indices.append(code_context.index('"' + tok + '"'))
				else:
					output_code_indices.append(code_context.index(tok))

			cur_data = {}
			cur_data['reserved_dfs'] = reserved_dfs
			cur_data['reserved_vars'] = reserved_vars
			cur_data['reserved_strs'] = reserved_strs
			cur_data['target_dfs'] = target_dfs
			cur_data['target_strs'] = target_strs
			cur_data['target_vars'] = target_vars
			cur_data['input_word_seq'] = input_word_seq
			cur_data['input_code_seq'] = input_code_seq
			cur_data['input_code_df_seq'] = input_code_df_seq
			cur_data['input_code_var_seq'] = input_code_var_seq
			cur_data['input_code_str_seq'] = input_code_str_seq
			cur_data['input_code_nl_indices'] = input_code_nl_indices
			cur_data['output_gt'] = output_gt
			cur_data['output_code_seq'] = output_code_seq
			cur_data['output_code_df_seq'] = output_code_df_seq
			cur_data['output_code_var_seq'] = output_code_var_seq
			cur_data['output_code_str_seq'] = output_code_str_seq
			cur_data['output_code_mask'] = output_code_mask
			cur_data['output_df_mask'] = output_df_mask
			cur_data['output_var_mask'] = output_var_mask
			cur_data['output_str_mask'] = output_str_mask
			cur_data['output_code_nl_indices'] = output_code_nl_indices
			cur_data['output_code_ctx_indices'] = output_code_ctx_indices
			cur_data['output_code_indices'] = output_code_indices
			cur_data['label'] = label
			data.append(cur_data)
			indices.append(sample_idx)
		print('Number of samples (before preprocessing): ', len(samples))
		print('Number of samples (after filtering): ', len(data))
		print('code seq len: min: ', min_target_code_seq_len, 'max: ', max_target_code_seq_len)
		return data, indices

	def get_batch(self, data, batch_size, start_idx):
		data_size = len(data)
		batch_vectors = []
		batch_labels = []
		batch_word_input = []
		batch_code_input = []
		batch_output_code_mask = []
		batch_output_df_mask = []
		batch_output_var_mask = []
		batch_output_str_mask = []
		batch_code_output = []
		batch_df_output = []
		batch_var_output = []
		batch_str_output = []
		batch_gt = []
		batch_input_code_nl_indices = []
		batch_output_code_nl_indices = []
		batch_output_code_ctx_indices = []
		input_dict = {}
		max_word_len = 0
		max_input_code_len = 0
		max_output_code_len = 0
		max_output_code_mask_len = 0
		if not self.copy_mechanism:
			max_output_code_mask_len = self.code_vocab_size + self.max_code_context_len
		for idx in range(start_idx, min(start_idx + batch_size, data_size)):
			cur_sample = data[idx]

			batch_word_input.append(cur_sample['input_word_seq'])
			max_word_len = max(max_word_len, len(cur_sample['input_word_seq']))
			batch_code_input.append(cur_sample['input_code_seq'])
			max_input_code_len = max(max_input_code_len, len(cur_sample['input_code_seq']))
			batch_output_code_mask.append(cur_sample['output_code_mask'])
			batch_output_df_mask.append(cur_sample['output_df_mask'])
			batch_output_var_mask.append(cur_sample['output_var_mask'])
			batch_output_str_mask.append(cur_sample['output_str_mask'])
			max_output_code_mask_len = max(max_output_code_mask_len, len(cur_sample['output_code_mask']))
			batch_gt.append(cur_sample['output_gt'])
			batch_code_output.append(cur_sample['output_code_seq'])
			max_output_code_len = max(max_output_code_len, len(cur_sample['output_code_seq']))
			batch_df_output.append(cur_sample['output_code_df_seq'])
			batch_var_output.append(cur_sample['output_code_var_seq'])
			batch_str_output.append(cur_sample['output_code_str_seq'])
			batch_input_code_nl_indices.append(cur_sample['input_code_nl_indices'])
			batch_output_code_nl_indices.append(cur_sample['output_code_nl_indices'])
			batch_output_code_ctx_indices.append(cur_sample['output_code_ctx_indices'])
			batch_labels.append(cur_sample['label'])

		batch_labels = np.array(batch_labels)
		batch_labels = np_to_tensor(batch_labels, 'int', self.cuda_flag)

		for idx in range(len(batch_word_input)):
			if len(batch_word_input[idx]) < max_word_len:
				batch_word_input[idx] = batch_word_input[idx] + [PAD_ID] * (max_word_len - len(batch_word_input[idx]))
		batch_word_input = np.array(batch_word_input)
		batch_word_input = np_to_tensor(batch_word_input, 'int', self.cuda_flag)
		input_dict['nl'] = batch_word_input
			
		for idx in range(len(batch_code_input)):
			if len(batch_code_input[idx]) < max_input_code_len:
				batch_code_input[idx] = batch_code_input[idx] + [PAD_ID] * (max_input_code_len - len(batch_code_input[idx]))
		batch_code_input = np.array(batch_code_input)
		batch_code_input = np_to_tensor(batch_code_input, 'int', self.cuda_flag)
		input_dict['code_context'] = batch_code_input

		for idx in range(len(batch_code_output)):
			if len(batch_code_output[idx]) < max_output_code_len:
				batch_code_output[idx] = batch_code_output[idx] + [PAD_ID] * (max_output_code_len - len(batch_code_output[idx]))
				batch_gt[idx] = batch_gt[idx] + [PAD_ID] * (max_output_code_len - len(batch_gt[idx]))
				batch_df_output[idx] = batch_df_output[idx] + [-1] * (max_output_code_len - len(batch_df_output[idx]))
				batch_var_output[idx] = batch_var_output[idx] + [-1] * (max_output_code_len - len(batch_var_output[idx]))
				batch_str_output[idx] = batch_str_output[idx] + [-1] * (max_output_code_len - len(batch_str_output[idx]))
		for idx in range(len(batch_output_code_mask)):
			if len(batch_output_code_mask[idx]) < max_output_code_mask_len:
				batch_output_code_mask[idx] = batch_output_code_mask[idx] + [0] * (max_output_code_mask_len - len(batch_output_code_mask[idx]))
				batch_output_df_mask[idx] = batch_output_df_mask[idx] + [0] * (max_output_code_mask_len - len(batch_output_df_mask[idx]))
				batch_output_var_mask[idx] = batch_output_var_mask[idx] + [0] * (max_output_code_mask_len - len(batch_output_var_mask[idx]))
				batch_output_str_mask[idx] = batch_output_str_mask[idx] + [0] * (max_output_code_mask_len - len(batch_output_str_mask[idx]))
		for idx in range(len(batch_input_code_nl_indices)):
			if len(batch_input_code_nl_indices[idx]) < max_input_code_len:
				batch_input_code_nl_indices[idx] = batch_input_code_nl_indices[idx] + [[max_word_len - 1, max_word_len - 1] for _ in range(max_input_code_len - len(batch_input_code_nl_indices[idx]))]

		batch_gt = np.array(batch_gt)
		batch_gt = np_to_tensor(batch_gt, 'int', self.cuda_flag)
		input_dict['gt'] = batch_gt
			
		batch_code_output = np.array(batch_code_output)
		batch_code_output = np_to_tensor(batch_code_output, 'int', self.cuda_flag)
		input_dict['code_output'] = batch_code_output

		batch_df_output = np.array(batch_df_output)
		batch_df_output = np_to_tensor(batch_df_output, 'int', self.cuda_flag)
		input_dict['df_output'] = batch_df_output

		batch_var_output = np.array(batch_var_output)
		batch_var_output = np_to_tensor(batch_var_output, 'int', self.cuda_flag)
		input_dict['var_output'] = batch_var_output

		batch_str_output = np.array(batch_str_output)
		batch_str_output = np_to_tensor(batch_str_output, 'int', self.cuda_flag)
		input_dict['str_output'] = batch_str_output

		batch_output_code_mask = np.array(batch_output_code_mask)
		batch_output_code_mask = np_to_tensor(batch_output_code_mask, 'float', self.cuda_flag)
		input_dict['code_output_mask'] = batch_output_code_mask

		batch_output_df_mask = np.array(batch_output_df_mask)
		batch_output_df_mask = np_to_tensor(batch_output_df_mask, 'float', self.cuda_flag)
		input_dict['output_df_mask'] = batch_output_df_mask

		batch_output_var_mask = np.array(batch_output_var_mask)
		batch_output_var_mask = np_to_tensor(batch_output_var_mask, 'float', self.cuda_flag)
		input_dict['output_var_mask'] = batch_output_var_mask

		batch_output_str_mask = np.array(batch_output_str_mask)
		batch_output_str_mask = np_to_tensor(batch_output_str_mask, 'float', self.cuda_flag)
		input_dict['output_str_mask'] = batch_output_str_mask

		batch_input_code_nl_indices = np.array(batch_input_code_nl_indices)
		batch_input_code_nl_indices = np_to_tensor(batch_input_code_nl_indices, 'int', self.cuda_flag)
		input_dict['input_code_nl_indices'] = batch_input_code_nl_indices
		batch_output_code_nl_indices = np.array(batch_output_code_nl_indices)
		batch_output_code_nl_indices = np_to_tensor(batch_output_code_nl_indices, 'int', self.cuda_flag)
		input_dict['output_code_nl_indices'] = batch_output_code_nl_indices
		batch_output_code_ctx_indices = np.array(batch_output_code_ctx_indices)
		batch_output_code_ctx_indices = np_to_tensor(batch_output_code_ctx_indices, 'int', self.cuda_flag)
		input_dict['output_code_ctx_indices'] = batch_output_code_ctx_indices
		input_dict['init_data'] = data[start_idx: start_idx + batch_size]
		return input_dict, batch_labels

