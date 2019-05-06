#! /usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf

from utils import const

__author__ = 'Jayeol Chun'


parser = argparse.ArgumentParser("DRSC Argparser")
parser.register("type", "bool", lambda v: v.lower() == "true")

parser.add_argument("--model_dir", required=True,
                    help="path to model output directory")

parser.add_argument("--model", type=str.lower, choices=const.MODELS,
                    help="which model to use")

parser.add_argument("--embedding", type=str, choices=const.EMBEDDINGS,
                    help="which embedding to use")

# preprocessing config
parser.add_argument(
  "--truncation_mode", type=str.lower, default='normal',
  choices=const.TRUNC_MODES,
  help="how to truncate tokens longer than `max_arg_length`")
parser.add_argument(
  "--do_lower_case", type="bool", nargs="?", const=True, default=True,
  help="whether to lower case the input text. Should be True for uncased model "
       "and False for cased model")
parser.add_argument(
  "--pooling_action", type=str.lower, default="sum",
  help="which pooling action to apply.")
parser.add_argument(
  "--do_pooling_first", type="bool", nargs="?", const=True, default=False,
  help="whether to apply pooling on word vectors (True) or on model outputs "
       "(False)")
parser.add_argument(
  # TODO: when extending to explicit types
  "--conn_action", type=str.lower, default=None,
  choices=const.CONN_ACTIONS, help="how to handle connectives")
parser.add_argument(
  "--padding_action", type=str.lower, default='normal',
  choices=const.POOLING_ACTIONS, help="how to pad up a batch")
parser.add_argument(
  "--use_one_hot_embeddings", type="bool", nargs="?", const=True, default=False,
  help="")

# embedding config
parser.add_argument(
  "--bert_model", type=str.lower, default="uncased",
  help="which bert model to use")

parser.add_argument(
  "--word_vector_width", type=int, default=768, help="dimension of word vector")
parser.add_argument(
  "--finetune_embedding", type="bool", nargs="?", const=True, default=False,
  help="whether to finetune embedding")

# model architecture config
parser.add_argument(
  "--hidden_size", type=int, default=768, help="hidden size of model layers")
parser.add_argument(
  "--num_hidden_layers", type=int, default=4,
  help="number of model's hidden layers")
parser.add_argument(
  "--num_attention_heads", type=int, default=8, help="attention head number")

# experimental setting config
parser.add_argument(
  "--num_epochs", type=int, default=3, help="how many iterations to train")
parser.add_argument(
  "--eval_every", type=int, default=100,
  help="how many batches per eval during training (-1 to disable)")
parser.add_argument(
  "--batch_size", type=int, default=64, help="batch size during training")
parser.add_argument(
  "--max_arg_length", type=int, default=128,
  help="how many tokens for each of arg to keep")
parser.add_argument(
  "--learning_rate", type=float, default=3e-4,
  help="learning rate during training")
parser.add_argument(
  "--optimizer", type=str.lower, default="adam", choices=const.OPTIMIZERS,
  help="which optimizer to use")

# sense-related config
parser.add_argument(
  "--sense_level", type=int, default=2, help="level of sense to use")
parser.add_argument(
  "--sense_type", type=str.lower, default="implicit", choices=const.SENSE_TYPES,
  help="which `Type` of data to use")
parser.add_argument(
  "--multiple_senses_action", type=str.lower, default="pick_first",
  choices=const.MULTIPLE_SENSES_ACTIONS,
  help="how to handle relations with multiple senses")
parser.add_argument(
  "--drop_partial_data", type="bool", nargs="?", const=True, default=False,
  help="whether to drop partial data")

# experiment control flags
parser.add_argument(
  "--do_train", type="bool", nargs="?", const=True, default=False,
  help="whether to run training")
parser.add_argument(
  "--do_eval", type="bool", nargs="?", const=True, default=False,
  help="whether to run eval")
parser.add_argument(
  "--do_predict", type="bool", nargs="?", const=True, default=False,
  help="whether to run predict")

# misc flags
parser.add_argument(
  "--allow_gpu_growth", type="bool", nargs="?", const=True, default=True,
  help="whether to allow gpu growth or pre-allocate all available resources")
parser.add_argument(
  "--do_debug", type="bool", nargs="?", const=True, default=False,
  help="whether to set logging verbosity to DEBUG")


def parse_args():
  return parser.parse_args()

def display_args(args):
  if isinstance(args, argparse.Namespace):
    args = vars(args)

  tf.logging.info("***** FLAGS *****")
  for k, v in args.items():
    tf.logging.info(f" {k}: {v}")
