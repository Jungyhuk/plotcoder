import argparse
import time
import os
import sys

def get_arg_parser(title):
	parser = argparse.ArgumentParser(description=title)
	parser.add_argument('--cpu', action='store_true', default=False)
	parser.add_argument('--eval', action='store_true')
	parser.add_argument('--model_dir', type=str, default='../checkpoints/model_0')
	parser.add_argument('--load_model', type=str, default=None)
	parser.add_argument('--num_LSTM_layers', type=int, default=2)
	parser.add_argument('--num_MLP_layers', type=int, default=1)
	parser.add_argument('--LSTM_hidden_size', type=int, default=512)
	parser.add_argument('--MLP_hidden_size', type=int, default=512)
	parser.add_argument('--embedding_size', type=int, default=512)

	parser.add_argument('--keep_last_n', type=int, default=None)
	parser.add_argument('--eval_every_n', type=int, default=1500)
	parser.add_argument('--log_interval', type=int, default=1500)
	parser.add_argument('--log_dir', type=str, default='../logs')
	parser.add_argument('--log_name', type=str, default='model_0.csv')

	parser.add_argument('--max_eval_size', type=int, default=1000)

	data_group = parser.add_argument_group('data')
	data_group.add_argument('--train_dataset', type=str, default='../data/train_plot.json')
	data_group.add_argument('--dev_dataset', type=str, default='../data/dev_plot_hard.json')
	data_group.add_argument('--test_dataset', type=str, default='../data/test_plot_hard.json')
	data_group.add_argument('--code_vocab', type=str, default='../data/code_vocab.json')
	data_group.add_argument('--word_vocab', type=str, default='../data/nl_vocab.json')
	data_group.add_argument('--word_vocab_size', type=int, default=None)
	data_group.add_argument('--code_vocab_size', type=int, default=None)
	data_group.add_argument('--num_plot_types', type=int, default=6)
	data_group.add_argument('--joint_plot_types', action='store_true', default=False)
	data_group.add_argument('--data_order_invariant', action='store_true', default=False)
	data_group.add_argument('--nl', action='store_true', default=False)
	data_group.add_argument('--use_comments', action='store_true', default=False)
	data_group.add_argument('--code_context', action='store_true', default=False)
	data_group.add_argument('--local_df_only', action='store_true', default=False)
	data_group.add_argument('--target_code_transform', action='store_true', default=False)
	data_group.add_argument('--max_num_code_cells', type=int, default=2)
	data_group.add_argument('--max_word_len', type=int, default=512)
	data_group.add_argument('--max_code_context_len', type=int, default=512)
	data_group.add_argument('--max_decode_len', type=int, default=200)

	model_group = parser.add_argument_group('model')
	model_group.add_argument('--hierarchy', action='store_true', default=False)
	model_group.add_argument('--copy_mechanism', action='store_true', default=False)
	model_group.add_argument('--nl_code_linking', action='store_true', default=False)

	train_group = parser.add_argument_group('train')
	train_group.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'])
	train_group.add_argument('--lr', type=float, default=1e-3)
	train_group.add_argument('--lr_decay_steps', type=int, default=6000)
	train_group.add_argument('--lr_decay_rate', type=float, default=0.9)
	train_group.add_argument('--dropout_rate', type=float, default=0.2)
	train_group.add_argument('--gradient_clip', type=float, default=5.0)
	train_group.add_argument('--num_epochs', type=int, default=50)
	train_group.add_argument('--batch_size', type=int, default=32)
	train_group.add_argument('--param_init', type=float, default=0.1)
	train_group.add_argument('--seed', type=int, default=None)

	return parser