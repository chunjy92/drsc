#! /usr/bin/python3
# -*- coding: utf-8 -*-
import random

import tensorflow as tf

import model
from data import PDTBProcessor
from embedding import Embedding

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, hp):
    self.hp = hp

    # init data preprocessor
    self.processor = PDTBProcessor()

    vocab = None
    if not self.hp.embedding:
      # if no embedding is specified, need to collect vocab from training set
      self.processor.compile_vocab_labels()
      vocab = self.processor.vocab
    else:
      self.processor.compile_labels()

    self.labels = self.processor.labels
    self.label_mapping = self.processor.get_label_mapping()

    # init embedding
    if self.hp.embedding == 'bert':
      raise NotImplementedError()
    else:
      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab)
      self.embedding_table = self.embedding.get_embedding_table()

    # init model
    self.build_model()

  def build_model(self):
    if self.hp.model == "mlp":
      self.model = model.MLP(labels=self.labels,
                             embedding_shape=self.embedding_table.shape,
                             finetune_embedding=False)

    tf.logging.info(f"{self.hp.model.upper()} model init")

  # TODO
  def load(self):
    pass

  def run(self):
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

  def batchify(self, examples, batch_size, do_shuffle=False):
    if do_shuffle:
      random.shuffle(examples)

    batches = []
    for start in range(0, len(examples), batch_size):
      # append the data from start pos of length batch size
      batches.append(examples[start:start+batch_size])
    return batches

  def train(self):
    examples = self.processor.get_train_examples()
    self.processor.remove_cache_by_key('train')

    init = tf.global_variables_initializer()

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess handled by Experiment obj, not by each model
    # self.sess = tf.Session(config=config)
    self.sess = tf.Session()

    tvars = tf.trainable_variables()
    tf.logging.info("***** Trainable Variables *****")
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    self.sess.run(
      [init, self.model.embedding_init_op],
      feed_dict={self.model.embedding_placeholder: self.embedding_table})

    for iter in range(1):
      feat_batches = self.batchify(examples, batch_size=64, do_shuffle=True)
      num_batches = len(feat_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      processed_batches = 0
      for i, batch in enumerate(feat_batches):
        batch = self.embedding.convert_to_ids(batch, self.label_mapping)

        arg1, arg2, conn, label = batch
        _, loss, acc = self.sess.run(
          [self.model.train_op, self.model.loss, self.model.acc],
          feed_dict={self.model.arg1: arg1,
                     self.model.arg2: arg2,
                     self.model.conn: conn,
                     self.model.label: label})
        processed_batches += 1
        tf.logging.info(
          "[{}th epoch {}/{} batches] loss: {:.3f} acc: {:.3f}".format(
            iter, processed_batches, num_batches, loss, acc))

        if (i+1) % 100 == 0:
          self.eval()

  def eval(self):
    examples = self.processor.get_dev_examples()
    examples = self.embedding.convert_to_ids(examples, self.label_mapping)
    arg1, arg2, conn, label = examples

    loss, acc = self.sess.run(
      [self.model.loss, self.model.acc], feed_dict={
        self.model.arg1: arg1,
        self.model.arg2: arg2,
        self.model.conn: conn,
        self.model.label: label
      }
    )

    tf.logging.info("[EVAL] loss: {:.3f} acc: {:.3f}".format(loss, acc))


  def predict(self):
    examples = self.processor.get_test_examples()
    self.processor.remove_cache_by_key('test')

    examples = self.embedding.convert_to_ids(examples, self.label_mapping)
    arg1, arg2, conn, label = examples

    acc = self.sess.run(
      [self.model.acc], feed_dict={
        self.model.arg1 : arg1,
        self.model.arg2 : arg2,
        self.model.conn : conn,
        self.model.label: label
      }
    )

    tf.logging.info("[PREDICT] acc: {:.3f}".format( acc))
