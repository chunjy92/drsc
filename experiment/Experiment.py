#! /usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import random
from abc import abstractmethod, ABC

import tensorflow as tf

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

    self.vocab = None
    if not self.hp.embedding:
      # if no embedding is specified, need to collect vocab from training set
      # TODO (May 5): Set a explicit flag value for random embedding, say,
      #  as `rand_init` or -
      self.processor.compile_vocab()
      self.vocab = self.processor.vocab

    self.labels = self.processor.labels
    tf.logging.info(f"All {len(self.labels)} Labels: {self.labels}")

    # label to index
    self.l2i = lambda l: self.labels.index(l)

    # index to label
    self.i2l = lambda i: self.labels[i]

    self.init_embedding()
    tf.logging.info("Embedding init")

    self.init_model()
    tf.logging.info("Model init")

  ############################ Absract methods #################################
  @abstractmethod
  def init_embedding(self):
    pass

  @abstractmethod
  def init_model(self):
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
  # TODO (April 27): is this really necessary? Currently not saving any model
  #   meta-data, including graphs and ckpts
  #  (May 5): necessary if tf.reset_default_graph(), but not at this point
  def load(self):
    raise NotImplementedError()

  def batchify(self, examples, batch_size, do_shuffle=False):
    if do_shuffle:
      random.shuffle(examples)

    batches = []
    for start in range(0, len(examples), batch_size):
      batch = examples[start:start + batch_size]
      batches.append(batch)

    return batches

  #################################### RUN #####################################
  def run(self):
    """Defines overall execution scheme consisting of 3 stages: train, eval and
    predict

    Each function has two versions: `{}_from_ids` and `{}_from_vals`.

    (1) `{}_from_ids` is when the loaded embedding is passed directly to the
        model, where values for each instance is retrieved through
        `tf.nn.embedding_lookup`.

    (2) `{}_from_vals` is when each data instance is converted into values
        before entering the TF computation graph, essentially equivalent to
        manual `tf.nn.embedding_lookup`. This is necessary when `self.embedding`
        is `BERTEmbedding` which will create a large output for each instance.
        Besides, each token will have different values in different contexts,
        so the conventional retrieval methods through id look-up doesn't work.
    """
    if self.hp.do_train:
      tf.logging.info("***** Begin Train *****")
      self.train()
    else:
      self.load()

    self.processor.remove_cache_by_key('train')

    if self.hp.do_eval:
      tf.logging.info("***** Begin Eval *****")
      self.eval()

    self.processor.remove_cache_by_key('dev')

    if self.hp.do_predict:
      tf.logging.info("***** Begin Predict *****")
      self.predict()
