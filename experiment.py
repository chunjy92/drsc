#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
import random
from collections import Counter

import numpy as np
import tensorflow as tf

import model
from data import PDTBProcessor
from embedding import BERTEmbedding, Embedding
from utils import const

__author__ = 'Jayeol Chun'


class Experiment(object):
  def __init__(self, hp):
    self.hp = hp

    # init data preprocessor
    self.processor = PDTBProcessor(
      max_arg_length=self.hp.max_arg_length,
      truncation_mode=self.hp.truncation_mode,
      do_lower_case=self.hp.do_lower_case,
      sense_type=self.hp.sense_type,
      sense_level=self.hp.sense_level,
      multiple_senses_action=self.hp.multiple_senses_action
    )

    self.vocab = None
    if not self.hp.embedding:
      # if no embedding is specified, need to collect vocab from training set
      self.processor.compile_vocab_labels()
      self.vocab = self.processor.vocab
    else:
      self.processor.compile_labels()

    self.labels = self.processor.labels

    # label to index
    self.l2i = lambda l: self.labels.index(l)

    # index to label
    self.i2l = lambda i: self.labels[i]

    self.init_embedding()
    tf.logging.info("Embedding init")

    self.init_model()
    tf.logging.info("Model init")

  ################################### INIT #####################################
  def init_embedding(self):
    if self.hp.embedding == 'bert':
      bert_model = const.BERT_TEMPLATE.format(self.hp.bert_model)
      bert_config_file = os.path.join(bert_model, const.BERT_CONFIG_FILE)
      bert_init_ckpt = os.path.join(bert_model, const.BERT_CKPT_FILE)
      bert_vocab_file = os.path.join(bert_model, const.BERT_VOCAB_FILE)

      self.embedding = BERTEmbedding(
        model_dir=self.hp.model_dir,
        bert_config_file=bert_config_file,
        vocab_file=bert_vocab_file,
        init_checkpoint=bert_init_ckpt,
        batch_size=self.hp.batch_size,
        max_arg_length=self.hp.max_arg_length,
        truncation_mode=self.hp.truncation_mode,
        do_lower_case=self.hp.do_lower_case,
        use_one_hot_embeddings=self.hp.use_one_hot_embeddings
      )

      self.hp.word_vector_width = 768
      self.embedding_table = None
      self.embedding_shape = None

    else:
      # reduce number of vocab and corresponding vector entries by collecting
      # all unique tokens in dev, test and train, which slows down
      vocab = self.processor.collect_all_vocab()
      tf.logging.info(f"All Vocab: {len(vocab)}")

      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab,
                                 max_arg_length=self.hp.max_arg_length)

      self.embedding_table = self.embedding.get_embedding_table()
      self.embedding_shape = self.embedding_table.shape

  def init_model(self):
    if self.hp.model == "mlp":
      self.model = model.MLP(
        labels=self.labels,
        max_arg_length=self.hp.max_arg_length,
        word_vector_width=self.hp.word_vector_width,
        hidden_size=self.hp.hidden_size,
        num_hidden_layers=self.hp.num_hidden_layers,
        learning_rate=self.hp.learning_rate,
        optimizer=self.hp.optimizer,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        embedding=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        do_pooling_first=self.hp.do_pooling_first,
        do_finetune_embedding=self.hp.do_finetune_embedding
      )
    else:
      # self.model = model.Attention(
      #   labels=self.labels,
      #   max_arg_length=self.hp.max_arg_length,
      #   word_vector_width=self.hp.word_vector_width,
      #   hidden_size=self.hp.hidden_size,
      #   num_hidden_layers=self.hp.num_hidden_layers,
      #   learning_rate=self.hp.learning_rate,
      #   optimizer=self.hp.optimizer,
      #   sense_type=self.hp.sense_type,
      #   pooling_action=self.hp.pooling_action,
      #   do_pooling_first=self.hp.do_pooling_first,
      # )
      self.model = model.Attention(self.labels)

  ################################### UTIL #####################################
  # TODO (April 27): is this really necessary? Currently not saving any model
  #   meta-data, including graphs and ckpts
  def load(self):
    pass

  def batchify(self, examples, batch_size, do_shuffle=False):
    if do_shuffle:
      random.shuffle(examples)

    batches = []
    for start in range(0, len(examples), batch_size):
      batch = examples[start:start+batch_size]
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

  ################################### TRAIN ####################################
  def train(self):
    """See docstring for `run`"""

    train_fn = None
    if self.hp.embedding == 'bert':
      examples = self.processor.get_train_examples(for_bert_embedding=True)
      train_fn = self.train_from_vals
    else:
      examples = self.processor.get_train_examples()
      train_fn = self.train_from_ids

    self.processor.remove_cache_by_key('train')

    self.init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    tf.logging.info("***** Trainable Variables *****")
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    config = tf.ConfigProto()

    # uncomment to pre-allocate all available GPU memory
    config.gpu_options.allow_growth = True

    # sess handled by Experiment obj, not by each model
    self.sess = tf.Session(config=config)

    train_fn(examples)

  def train_from_vals(self, examples):
    bert_outputs = self.embedding.run(examples)

    # coupling of these in prep for random shuffling
    bert_examples_pair = \
      [(example, bert_output)
       for example, bert_output in zip(examples, bert_outputs)]

    self.sess.run(self.init)

    for iter in range(self.hp.num_iter):
      bert_batches = self.batchify(bert_examples_pair,
                                   batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(bert_batches)

      processed_batches = 0
      for i, batch in enumerate(bert_batches):

        # tedious decoupling
        label_ids = []
        batch_bert_outputs = []
        for batch_example, batch_bert_output in batch:
          batch_bert_outputs.append(batch_bert_output)

          # label
          label_ids.append(self.l2i(batch_example.label))

        # prepare bert output: [batch, total_seq_length, bert_hidden_size]
        batch_bert_outputs = np.asarray(batch_bert_outputs)
        total_seq_length = batch_bert_outputs.shape[1]
        assert total_seq_length == self.hp.max_arg_length * 2, \
          "Sequence length mismatch between BERT output and parameter"

        arg1 = batch_bert_outputs[:,:self.hp.max_arg_length,:]
        arg2 = batch_bert_outputs[:,self.hp.max_arg_length:,:]

        # connectives
        # since we don't care about connectives, make them 0 for now
        conn = np.zeros([self.hp.batch_size,
                         self.hp.max_arg_length,
                         batch_bert_outputs.shape[-1]])

        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss,
           self.model.acc],
          feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                     self.model.conn: conn, self.model.label: label_ids})
        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(self.labels[key])
        processed_batches += 1
        tf.logging.info(
          "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f} {}".format(
            iter, processed_batches, num_batches, loss, acc, c))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

  def train_from_ids(self, examples, l2_reg_weights=None):
    self.sess.run(
      [self.init, self.model.embedding_init_op],
      feed_dict={self.model.embedding_placeholder: self.embedding_table})

    for iter in range(self.hp.num_iter):
      feat_batches = self.batchify(examples, batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(feat_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      processed_batches = 0
      for i, batch in enumerate(feat_batches):
        batch = self.embedding.convert_to_ids(batch, self.l2i)

        arg1, arg2, conn, label_ids = batch
        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss, self.model.acc],
          feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                     self.model.conn: conn, self.model.label: label_ids})
        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(self.labels[key])
        processed_batches += 1
        tf.logging.info(
          "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f} {}".format(
            iter, processed_batches, num_batches, loss, acc, c))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

  ################################### EVAL #####################################
  def eval(self):
    """See docstring for `run`"""

    eval_fn = None
    if self.hp.embedding == 'bert':
      examples = self.processor.get_dev_examples(for_bert_embedding=True)
      eval_fn = self.eval_from_vals
    else:
      examples = self.processor.get_dev_examples()
      eval_fn = self.eval_from_ids

    eval_fn(examples)

  def eval_from_vals(self, examples):
    bert_outputs = self.embedding.run(examples)

    # prepare bert output: [batch, total_seq_length, bert_hidden_size]
    bert_outputs = np.asarray(bert_outputs)
    total_seq_length = bert_outputs.shape[1]
    assert total_seq_length == self.hp.max_arg_length * 2, \
      "Sequence length mismatch between BERT output and parameter"

    arg1 = bert_outputs[:, :self.hp.max_arg_length, :]
    arg2 = bert_outputs[:, self.hp.max_arg_length:, :]

    label_ids = []
    for example in examples:
      label_ids.append(self.l2i(example.label))

    # connectives
    # since we don't care about connectives, make them 0 for now
    conn = np.zeros([self.hp.batch_size,
                     self.hp.max_arg_length,
                     bert_outputs.shape[-1]])

    loss, preds, acc = self.sess.run(
      [self.model.loss, self.model.preds, self.model.acc],
      feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                 self.model.conn: conn, self.model.label: label_ids})

    c = Counter(preds)
    tf.logging.info("[EVAL] loss: {:.3f} acc: {:.3f} {}".format(loss, acc, c))

  def eval_from_ids(self, examples):

    examples = self.embedding.convert_to_ids(examples, self.l2i)
    arg1, arg2, conn, label = examples

    loss, preds, acc = self.sess.run(
      [self.model.loss, self.model.preds, self.model.acc],
      feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                 self.model.conn: conn, self.model.label: label})

    c = Counter(preds)
    tf.logging.info("[EVAL] loss: {:.3f} acc: {:.3f} {}".format(loss, acc, c))

  ################################## PREDICT ###################################
  def predict(self):
    """See docstring for `run`"""

    for i in range(2):

      dataset_type = "test" if i==0 else "blind"

      predict_fn = None
      if self.hp.embedding == 'bert':
        if i==0:
          examples = self.processor.get_test_examples(for_bert_embedding=True)
        else:
          examples = self.processor.get_blind_examples(for_bert_embedding=True)
        predict_fn = self.predict_from_vals
      else:
        if i==0:
          examples = self.processor.get_test_examples()
        else:
          examples = self.processor.get_blind_examples()
        predict_fn = self.predict_from_ids

      tf.logging.info(f"Inference on {dataset_type} set")
      self.processor.remove_cache_by_key(dataset_type)
      preds = predict_fn(examples)

      preds_file = \
        os.path.join(self.hp.model_dir, f"{dataset_type}_predictions.txt")

      tf.logging.info(f"Exporting test predictions at {preds_file}")
      with open(preds_file, 'w') as f:
        f.write("\n".join(preds))

  def predict_from_vals(self, examples):
    bert_outputs = self.embedding.run(examples)

    # prepare bert output: [batch, total_seq_length, bert_hidden_size]
    bert_outputs = np.asarray(bert_outputs)
    total_seq_length = bert_outputs.shape[1]
    assert total_seq_length == self.hp.max_arg_length * 2, \
      "Sequence length mismatch between BERT output and parameter"

    arg1 = bert_outputs[:, :self.hp.max_arg_length, :]
    arg2 = bert_outputs[:, self.hp.max_arg_length:, :]

    label_ids = []
    for example in examples:
      label_ids.append(self.l2i(example.label))

    # connectives
    # since we don't care about connectives, make them 0 for now
    conn = np.zeros([self.hp.batch_size,
                     self.hp.max_arg_length,
                     bert_outputs.shape[-1]])

    preds, acc = self.sess.run(
      [self.model.preds, self.model.acc],
      feed_dict={self.model.arg1: arg1, self.model.arg2: arg2,
                 self.model.conn: conn, self.model.label: label_ids})

    c = Counter(preds)
    tf.logging.info("[PREDICT] acc: {:.3f} {}".format(acc, c))

    # convert label_id preds to labels
    preds_str = [self.i2l(pred) for pred in preds]

    return preds_str

  def predict_from_ids(self, examples):
    self.processor.remove_cache_by_key('test')

    examples = self.embedding.convert_to_ids(examples, self.l2i)
    arg1, arg2, conn, label = examples

    preds, acc = self.sess.run(
      [self.model.preds, self.model.acc],
      feed_dict={self.model.arg1 : arg1, self.model.arg2 : arg2,
                 self.model.conn : conn, self.model.label: label})

    c = Counter(preds)
    tf.logging.info("[PREDICT] acc: {:.3f} {}".format(acc, c))

    # convert label_id preds to labels
    preds_str = [self.i2l(pred) for pred in preds]
    return preds_str
