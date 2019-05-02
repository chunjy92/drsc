#! /usr/bin/python3
# -*- coding: utf-8 -*-
import time

import tensorflow as tf

from experiment import DRSCExperiment
from utils import config, logging

__author__ = 'Jayeol Chun'


FLAGS = None

def main(_):
  begin = time.time()

  tf.gfile.MakeDirs(FLAGS.model_dir)

  # redirects tf logs to file
  logging.init_logger(FLAGS.model_dir, FLAGS.do_debug)
  config.display_args(FLAGS)

  E = DRSCExperiment(FLAGS)
  E.run()

  tf.logging.info("Execution Time: {:.2f}s".format(time.time() - begin))

if __name__ == '__main__':
  FLAGS = config.parse_args()
  tf.app.run()
