#! /usr/bin/python3
# -*- coding: utf-8 -*-
import time

import tensorflow as tf

import utils

__author__ = 'Jayeol Chun'


FLAGS = None

def main(_):
  begin = time.time()

  tf.gfile.MakeDirs(FLAGS.model_dir)

  # redirects tf logs to file
  utils.init_logger(FLAGS.model_dir)
  utils.display_args(FLAGS)

  tf.logging.info("Execution Time: {:.2f}s".format(time.time() - begin))

if __name__ == '__main__':
  FLAGS = utils.parse_args()
  tf.app.run()
