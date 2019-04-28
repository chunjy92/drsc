#! /usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf

from utils import const

__author__ = 'Jayeol Chun'


parser = argparse.ArgumentParser("DRSC Argparser")

parser.add_argument("--model_dir", required=True,
                    help="path to model output directory")

parser.add_argument("--model", choices=const.MODELS,
                    help="which model to use")

parser.add_argument("--embedding", choices=const.EMBEDDINGS,
                    help="which embedding to use")

parser.add_argument(
  "--do_train", action="store_true", help="whether to run training")
parser.add_argument(
  "--do_eval", action="store_true", help="whether to run eval")
parser.add_argument(
  "--do_predict", action="store_true", help="whether to run predict")


def parse_args():
  return parser.parse_args()

def display_args(args):
  if isinstance(args, argparse.Namespace):
    args = vars(args)

  tf.logging.info("***** FLAGS *****")
  for k, v in args.items():
    tf.logging.info(f" {k}: {v}")
