#! /usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import os
import random
from abc import abstractmethod, ABC

import tensorflow as tf

from bert import modeling
from data import PDTBProcessor

__author__ = 'Jayeol Chun'


class Experiment(ABC):
  def __init__(self, hp):
    self.hp = copy.deepcopy(hp)

    # init data preprocessor
    self.processor = PDTBProcessor(
      max_arg_length=self.hp.max_arg_length,
      truncation_mode=self.hp.truncation_mode,
      do_lower_case=self.hp.do_lower_case,
      sense_type=self.hp.sense_type,
      sense_level=self.hp.sense_level,
      multiple_senses_action=self.hp.multiple_senses_action,
      padding_action=self.hp.padding_action,
      drop_partial_data=self.hp.drop_partial_data
    )

    self.labels = self.processor.labels
    tf.logging.info(f"All {len(self.labels)} Labels: {self.labels}")

    # label to index
    self.l2i = lambda l: self.labels.index(l)

    # index to label
    self.i2l = lambda i: self.labels[i]

    # save/restoring
    self.model_ckpt_path = os.path.join(self.hp.model_dir, "model.ckpt")

    # sess config
    self.sess_config = tf.ConfigProto()

    # if False,pre-allocates all available GPU memory
    self.sess_config.gpu_options.allow_growth = self.hp.allow_gpu_growth

    # if not is_training:
    self.num_train_steps = 0
    self.num_warmup_steps = 0

  #############################  Absract methods ###############################
  @abstractmethod
  def init_embedding(self, is_training=False):
    pass

  @abstractmethod
  def init_model(self, is_training=False):
    pass

  @abstractmethod
  def train(self):
    pass

  @abstractmethod
  def eval(self):
    pass

  @abstractmethod
  def predict(self):
    pass

  ################################### UTIL #####################################
  def init_bert_from_checkpoint(self):
    tvars = tf.trainable_variables()

    if self.hp.embedding == "bert":
      initialized_variable_names = []
      (assignment_map, initialized_variable_names) = \
        modeling.get_assignment_map_from_checkpoint(
          tvars, self.embedding.init_checkpoint)

      tf.train.init_from_checkpoint(
        self.embedding.init_checkpoint, assignment_map)

      tf.logging.info("**** Trainable Variables ****")
      for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
          init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name,
                        var.shape, init_string)

  def batchify(self, examples, batch_size, do_shuffle=False):
    if do_shuffle:
      random.shuffle(examples)

    batches = []
    for start in range(0, len(examples), batch_size):
      batch = examples[start:start+batch_size]
      batches.append(batch)

    return batches

  def display_log(self, pred_counter, msg_header, mean_loss=None,
                  mean_acc=None):
    msg = msg_header

    if mean_loss is not None:
      msg += " loss: {:.3f}".format(mean_loss)

    if mean_acc is not None:
      msg += " acc: {:.3f}".format(mean_acc)

    tf.logging.info(msg)
    for key in sorted(pred_counter.keys()):
      tf.logging.info(" {:4d}: {}".format(pred_counter[key], self.labels[key]))

  #################################### RUN #####################################
  def run(self):
    """Defines overall execution scheme consisting of 3 stages: train, eval and
    predict
    """
    if self.hp.do_train:
      self.train()
      self.processor.remove_cache_by_key('train')

    # `eval` called from `train` at every end of epoch
    if self.hp.do_eval and not self.hp.do_train:
      self.eval()
      self.processor.remove_cache_by_key('dev')

    if self.hp.do_predict:
      self.predict()
