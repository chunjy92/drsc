#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
import time

import tensorflow as tf

from experiment import DRSCExperiment
from utils import config, const, logging

__author__ = 'Jayeol Chun'


FLAGS = None

def run_bert_classifier(log_file):
  import sys
  import subprocess

  pythonpath = "PYTHONPATH=."
  python = "python"
  bertclf = "model/BERTClassifier.py"
  command = f"{pythonpath} {python} {bertclf}"
  command += " --data_dir=./data"
  command += " --task_name=drsc"

  # some translation between our FLAGS and their FLAGS
  for k, v in vars(FLAGS).items():
    if k == 'model_dir':
      k = "output_dir"
      command += f" --{k}={v}"

    if k == 'bert_model':
      bert_path = const.BERT_TEMPLATE
      if v == "uncased":
        bert_path = bert_path.format('uncased')

      else:  # v == 'cased'
        bert_path = bert_path.format('cased')

      bert_vocab = os.path.join(bert_path, const.BERT_VOCAB_FILE)
      bert_config = os.path.join(bert_path, const.BERT_CONFIG_FILE)
      bert_ckpt = os.path.join(bert_path, const.BERT_CKPT_FILE)

      command += f" --vocab_file={bert_vocab}"
      command += f" --bert_config_file={bert_config}"
      command += f" --init_checkpoint={bert_ckpt}"

    if k=='max_arg_length':
      k = 'max_seq_length'
      command += f" --{k}={v}"

    if k=='num_epochs':
      k = 'num_train_epochs'
      command += f" --{k}={v}."

    if k=='batch_size':
      k = 'train_batch_size'
      command += f" --{k}={v}"

    if k in ["do_lower_case", "drop_partial_data", "learning_rate",
             "do_train", "do_eval", "do_predict"]:
      command += f" --{k}={v}"

  log_file = open(log_file, 'a')

  proc = subprocess.Popen(
    command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True,
    universal_newlines=True)

  for line in proc.stdout:
    sys.stdout.write(line)
    log_file.write(line)

  proc.wait()
  log_file.close()

def main(_):
  begin = time.time()

  tf.gfile.MakeDirs(FLAGS.model_dir)

  # redirects tf logs to file
  log_file = logging.init_logger(FLAGS.model_dir, FLAGS.do_debug)
  config.display_args(FLAGS)

  if FLAGS.model == "bert":
    run_bert_classifier(log_file)
  else:
    E = DRSCExperiment(FLAGS)
    E.run()

  tf.logging.info("Execution Time: {:.2f}s".format(time.time() - begin))

if __name__ == '__main__':
  FLAGS = config.parse_args()
  tf.app.run()
