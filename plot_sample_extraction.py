import argparse
import math
import random
import sys
import os
import json
import numpy as np
import time
import operator


scatter_word_list = ['scatter', "'scatter'", '"scatter"', 'scatter_kws', "'o'", "'bo'", "'r+'", '"o"', '"bo"', '"r+"']
hist_word_list = ['hist', "'hist'", '"hist"', 'bar', "'bar'", '"bar"', 'countplot', 'barplot']
pie_word_list = ['pie', "'pie'", '"pie"']
scatter_plot_word_list = ['lmplot', 'regplot']
hist_plot_word_list = ['distplot', 'kdeplot', 'contour']
normal_plot_word_list = ['plot']

reserved_words = scatter_word_list + hist_word_list + pie_word_list + scatter_plot_word_list + hist_plot_word_list + normal_plot_word_list


arg_parser = argparse.ArgumentParser(description='JuiCe plot data extraction')
arg_parser.add_argument('--data_folder', type=str, default='../data',
	help="the folder where the datasets downloaded from the original JuiCe repo are stored. We will retrieve 'train.jsonl', 'dev.jsonl' and 'test.jsonl' here.")
arg_parser.add_argument('--init_train_data_name', type=str, default='train.jsonl',
	help="the filename of the original training data.")
arg_parser.add_argument('--init_dev_data_name', type=str, default='dev.jsonl',
	help="the filename of the original dev data.")
arg_parser.add_argument('--init_test_data_name', type=str, default='test.jsonl',
	help="the filename of the original test data.")
arg_parser.add_argument('--prep_train_data_name', type=str, default='train_plot.json',
	help="the filename of the preprocessed training data. When set to None, it means that the training data is not preprocessed (this file is the most time-consuming for preprocessing).")
arg_parser.add_argument('--prep_dev_data_name', type=str, default='dev_plot.json',
	help="the filename of the preprocessed dev data. When set to None, it means that the dev data is not preprocessed.")
arg_parser.add_argument('--prep_test_data_name', type=str, default='test_plot.json',
	help="the filename of the preprocessed test data. When set to None, it means that the test data is not preprocessed.")
arg_parser.add_argument('--prep_dev_hard_data_name', type=str, default='dev_plot_hard.json',
	help="the filename of the preprocessed hard split of the dev data. When set to None, it means that the dev data is not preprocessed.")
arg_parser.add_argument('--prep_test_hard_data_name', type=str, default='test_plot_hard.json',
	help="the filename of the preprocessed hard split of the test data. When set to None, it means that the test data is not preprocessed.")
arg_parser.add_argument('--build_vocab', action='store_true', default=True,
	help="set the flag to be true, so as to build the natural language word and code vocabs from the training set.")
arg_parser.add_argument('--nl_freq_file', type=str, default='nl_freq.json',
	help='the file that stores the frequency of each natural language word.')
arg_parser.add_argument('--code_freq_file', type=str, default='code_freq.json',
	help='the file that stores the frequency of each code token.')
arg_parser.add_argument('--nl_vocab', type=str, default='nl_vocab.json',
	help='the file that stores the natural language word vocabulary.')
arg_parser.add_argument('--code_vocab', type=str, default='code_vocab.json',
	help='the file that stores the code token vocabulary.')
arg_parser.add_argument('--min_nl_freq', type=int, default=15,
	help='Words with a smaller number of occurrences in the training data than this threshold are excluded from the nl word vocab.')
arg_parser.add_argument('--min_code_freq', type=int, default=1000,
	help='Code tokens with a smaller number of occurrences in the training data than this threshold are excluded from the code token vocab.')

args = arg_parser.parse_args()


def preprocess(data_folder, init_data_name, prep_data_name, prep_hard_data_name=None, additional_samples=[], is_train=True):
	plot_samples = []
	clean_samples = []
	init_data_name = os.path.join(data_folder, init_data_name)
	with open(init_data_name) as fin:
		for i, line in enumerate(fin):
			sample = json.loads(line)

			# extract code sequence without comments and empty strings
			init_code_seq = sample['code_tokens']
			code_seq = []
			for tok in init_code_seq:
				if len(tok) == 0 or tok[0] == '#':
					continue
				code_seq.append(tok)

			# filter out samples where 'plt' is not used
			while 'plt' in code_seq:
				pos = code_seq.index('plt')
				if pos < len(code_seq) - 1 and code_seq[pos + 1] == '.':
					break
				code_seq = code_seq[pos + 1:]
			if not ('plt' in code_seq):
				continue

			plot_calls = []
			api_seq = sample['api_sequence']
			for api in api_seq:
				if api == 'subplot':
					continue
				if api[-4:] == 'plot' and not ('_' in api):
					plot_calls.append(api)

			exist_plot_calls = False
			for code_idx, tok in enumerate(code_seq):
				if not (tok in reserved_words + plot_calls):
					continue
				if code_idx == len(code_seq) - 1 or code_seq[code_idx + 1] != '(':
					continue
				exist_plot_calls = True
				break
			if not exist_plot_calls:
				continue

			url = sample['metadata']['path']
			if 'solution' in url.lower() or 'assignment' in url.lower():
				clean_samples.append(sample)
				if not is_train:
					plot_samples.append(sample)
			else:
				plot_samples.append(sample)

	print('number of samples in the original partition: ', len(plot_samples))
	print('number of course-related samples in the partition: ', len(clean_samples))
	json.dump(plot_samples, open(os.path.join(data_folder, prep_data_name), 'w'))
	if len(additional_samples) > 0:
		print('number of samples in the hard partition: ', len(additional_samples))
		json.dump(additional_samples, open(os.path.join(data_folder, prep_hard_data_name), 'w'))
	return plot_samples, clean_samples


def add_token_to_dict(seq, vocab_dict, is_code=False):
	for tok in seq:
		if len(tok) == 0:
			continue
		if is_code and tok[0] == '#':
			continue
		if tok in vocab_dict:
			vocab_dict[tok] += 1
		else:
			vocab_dict[tok] = 1
	return vocab_dict

def build_vocab(samples):

	# Compute the frequency of each nl and code token
	code_dict = {}
	word_dict = {}

	for sample in samples:
		context = sample['context']
		for cell in context:
			if not 'code_tokens' in cell:
				continue
			code_context = cell['code_tokens']
			if type(code_context) != list:
				continue
			code_dict = add_token_to_dict(code_context, code_dict, is_code=True)
		code_dict = add_token_to_dict(sample['code_tokens'], code_dict, is_code=True)
		word_dict = add_token_to_dict(sample['nl'] + sample['comments'], word_dict, is_code=False)

	sorted_word_list = sorted(word_dict.items(), key=operator.itemgetter(1), reverse=True)
	sorted_code_list = sorted(code_dict.items(), key=operator.itemgetter(1), reverse=True)
	print('Total number of nl tokens (before filtering): ', len(sorted_word_list))
	print('Total number of code tokens (before filtering): ', len(sorted_code_list))
	json.dump(sorted_word_list, open(os.path.join(args.data_folder, args.nl_freq_file), 'w'))
	json.dump(sorted_code_list, open(os.path.join(args.data_folder, args.code_freq_file), 'w'))

	# filter out rare tokens
	code_vocab = {}
	word_vocab = {}

	for i, item in enumerate(sorted_word_list):
		if item[1] < args.min_nl_freq:
			break
		word_vocab[item[0]] = i

	for i, item in enumerate(sorted_code_list):
		if item[1] < args.min_code_freq:
			break
		code_vocab[item[0]] = i

	print('Total number of nl tokens (after filtering): ', len(word_vocab))
	print('Total number of code tokens (after filtering): ', len(code_vocab))
	json.dump(word_vocab, open(os.path.join(args.data_folder, args.nl_vocab), 'w'))
	json.dump(code_vocab, open(os.path.join(args.data_folder, args.code_vocab), 'w'))


if not os.path.exists(args.data_folder):
	os.makedirs(args.data_folder)

# data preprocessing
if args.prep_train_data_name:
	print('preprocessing training data:')
	train_plot_samples, train_plot_clean_samples = preprocess(args.data_folder, args.init_train_data_name, args.prep_train_data_name, is_train=True)
	cnt_train_clean_samples = len(train_plot_clean_samples)

if args.prep_dev_data_name:
	print('preprocessing dev data:')
	dev_plot_samples, dev_plot_clean_samples = preprocess(args.data_folder, args.init_dev_data_name, args.prep_dev_data_name,
		prep_hard_data_name=args.prep_dev_hard_data_name, additional_samples=train_plot_clean_samples[:cnt_train_clean_samples // 2], is_train=False)

if args.prep_test_data_name:
	print('preprocessing test data:')
	test_plot_samples, test_plot_clean_samples = preprocess(args.data_folder, args.init_test_data_name, args.prep_test_data_name,
		prep_hard_data_name=args.prep_test_hard_data_name, additional_samples=train_plot_clean_samples[cnt_train_clean_samples // 2:], is_train=False)

# build natural language word and code vocabularies
if args.build_vocab:
	assert args.init_train_data_name is not None
	build_vocab(train_plot_samples)