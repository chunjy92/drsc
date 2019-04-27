#! /usr/bin/python3
# -*- coding: utf-8 -*-
import argparse

import tensorflow as tf

__author__ = 'Jayeol Chun'


parser = argparse.ArgumentParser("DRSC Argparser")

parser.add_argument("--model_dir", required=True,
                    help="path to model output directory")


def parse_args():
  return parser.parse_args()

def display_args(args):
  if isinstance(args, argparse.Namespace):
    args = vars(args)

  tf.logging.info("***** FLAGS *****")
  for k, v in args.items():
    tf.logging.info(f" {k}: {v}")
