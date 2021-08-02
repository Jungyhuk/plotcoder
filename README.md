# PlotCoder: Hierarchical Decoding for Synthesizing Visualization Code in Programmatic Context

This repo provides the code to replicate the experiments in the [paper](https://aclanthology.org/2021.acl-long.169/)

> Xinyun Chen, Linyuan Gong, Alvin Cheung, Dawn Song, <cite> PlotCoder: Hierarchical Decoding for Synthesizing Visualization Code in Programmatic Context, in ACL 2021. </cite>

## Prerequisites

Download the JuiCe dataset [here](https://github.com/rajasagashe/juice).

The code is runnable with Python 3, PyTorch 0.4.1.

## Data preprocessing

To extract the plot generation samples from the entire JuiCe dataset, run ``plot_sample_extraction.py``.

Note that for model training and evaluation, we may further filter out some samples from the datasets extracted here. But we would like to keep these copies that include the maximal number of plot samples we can extract, so that we no longer need to enumerate the entire JuiCe dataset afterwards.

### Key arguments

In the following we list some important arguments for data preprocessing:
* `--data_folder`: path to the directory that stores the data.
* `--prep_train_data_name`: filename of the plot generation samples for training. Note that it does not include all plot samples in the original JuiCe training split: some of them are merged into the hard dev/test splits. To build the training set, make sure that this filename is not `None`.
* `--prep_dev_data_name`, `--prep_test_data_name`: filename of the plot generation samples filtered from the original dev/test splits of JuiCe. To preprocess each split of the data, make sure that the corresponding filename is not `None`.
* `--prep_dev_hard_data_name`, `--prep_test_hard_data_name`: filename of the homework or exam solutions extracted from the original training split of JuiCe. These are larger-scale sets for evaluation.
* `--build_vocab`: set it to be `True` for building the vocabularies of natural language words and code tokens.

## Run experiments

1. To run the hierarchical model:

`python run.py --nl --use_comments --code_context --nl_code_linking --copy_mechanism --hierarchy --target_code_transform`

2. To run the non-hierarchical model with the copy mechanism:

`python run.py --nl --use_comments --code_context --nl_code_linking --copy_mechanism  --target_code_transform`

3. To run the LSTM decoder without the copy mechanism, i.e., one-hot encoding for data items, but preserve the nl correspondence in the input code sequence:

`python run.py --nl --use_comments --code_context --nl_code_linking  --target_code_transform`

4. To run the LSTM decoder with the standard copy mechanism, do not preserve the nl correspondence:

`python run.py --nl --use_comments --code_context --copy_mechanism  --target_code_transform`

5. To run the LSTM decoder without the copy mechanism, i.e., one-hot encoding for data items as in prior work:

`python run.py --nl --use_comments --code_context  --target_code_transform`

### Key arguments
In the following we list some important arguments for running neural models:
* `--nl`: include the previous natural language cell as the model input. Note that the current code does not support including natural language from multiple cells, because it may not make sense to add NL instructions for previous code cells instead of the current one to confuse the model.
* `--use_comments`: include the comments in the current code cell as the model input.
* `--code_context`: include the code context as the model input.
* `--target_code_transform`: standardize the target code sequence into a more canonical form.
* `--max_num_code_cells`: the number of code cells included as the code context. Default: `2`. Note that setting it to `0` is not equivalent to not using the code context, because it still includes: (1) the code within the current code cell before the code snippet starting to generate the plots; and (2) the code context including the data frames and their attributes.
* `--nl_code_linking`: if a code token appears in the nl, concatenate the code token embedding with the corresponding nl embedding.
* `--copy_mechanism`: use the copy mechanism for the decoder.
* `--hierarchy`: use the hierarchical decoder for code generation.
* `--load_model`: path to the trained model (not required when training from scratch).
* `--eval`: add this command during the test time, and remember to set `--load_model` for evaluation.

## Citation

If you use the code in this repo, please cite the following paper:

```
@inproceedings{chen-2021-plotcoder,
    title={PlotCoder: Hierarchical Decoding for Synthesizing Visualization Code in Programmatic Context},
    author={Chen, Xinyun  and
      Gong, Linyuan  and
      Cheung, Alvin  and
      Song, Dawn},
    booktitle={Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)},
    year={2021}
}
```