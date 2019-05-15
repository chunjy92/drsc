#! /usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf

from utils import const

__author__ = 'Jayeol Chun'


parser = argparse.ArgumentParser("DRSC Argparser")
parser.register("type", "bool", lambda v: v.lower() == "true")

# REQUIRED config
parser.add_argument("--model_dir", required=True,
                    help="path to model output directory")

parser.add_argument("--model", type=str.lower, default='mask_attn',
                    choices=const.MODELS, help="which model to use")

parser.add_argument("--embedding", type=str, default='bert',
                    choices=const.EMBEDDINGS, help="which embedding to use")

parser.add_argument("--attention", type=str.lower, default='self_attn',
                    choices=const.ATTENTIONS,
                    help="which attentional body to use")

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
  choices=const.POOLING_ACTIONS, help="which pooling action to apply.")

# embedding config
parser.add_argument(
  "--bert_model", type=str.lower, default="uncased",
  help="which bert model to use")
parser.add_argument(
  "--word_vector_width", type=int, default=768, help="dimension of word vector")
parser.add_argument(
  "--finetune_embedding", type="bool", nargs="?", const=True, default=False,
  help="whether to finetune embedding")
parser.add_argument(
  "--split_args_in_embedding", type="bool", nargs="?", const=True, default=True,
  help="whether to treat Arg1 and Arg2 separately in bert computation")

# model architecture config
parser.add_argument(
  "--hidden_size", type=int, default=512, help="hidden size of model layers")
parser.add_argument(
  "--num_hidden_layers", type=int, default=4,
  help="number of model's hidden layers")
parser.add_argument(
  "--num_attention_heads", type=int, default=8, help="attention head number")

# experimental setting config
parser.add_argument(
  "--num_epochs", type=int, default=5, help="how many iterations to train")
parser.add_argument(
  "--log_every", type=int, default=32,
  help="how many batch iters per logging")
parser.add_argument(
  "--batch_size", type=int, default=32, help="batch size during training")
parser.add_argument(
  "--max_seq_length", type=int, default=128,
  help="how many tokens to keep")
parser.add_argument(
  "--learning_rate", type=float, default=5e-5,
  help="learning rate during training")
parser.add_argument(
    "--warmup_proportion", type=float, default=0.1,
    help="Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")
parser.add_argument(
  "--optimizer", type=str.lower, default="adam", choices=const.OPTIMIZERS,
  help="which optimizer to use")

# control actions config
parser.add_argument(
  "--do_pooling_first", type="bool", nargs="?", const=True, default=True,
  help="whether to apply pooling on word vectors (True) or on model outputs "
       "(False). Not used in `Attentional` model")
parser.add_argument(
  "--cls_action", type=str.lower, default="first_cls",
  choices=const.CLS_ACTIONS, help="which cls pooling action should be applied")
parser.add_argument(
  # TODO: when extending to explicit types
  #   (May 11) Currently not used
  "--conn_action", type=str.lower, default=None,
  choices=const.CONN_ACTIONS, help="how to handle connectives")
parser.add_argument(
  "--padding_action", type=str.lower, default='normal',
  choices=const.PADDING_ACTIONS, help="how to pad up a batch")

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
  "--do_train", type="bool", nargs="?", const=True, default=True,
  help="whether to run training")
parser.add_argument(
  "--do_eval", type="bool", nargs="?", const=True, default=True,
  help="whether to run eval")
parser.add_argument(
  "--do_predict", type="bool", nargs="?", const=True, default=True,
  help="whether to run predict")

# misc flags
parser.add_argument(
  "--allow_gpu_growth", type="bool", nargs="?", const=True, default=False,
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
