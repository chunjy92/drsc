#! /usr/bin/python3
# -*- coding: utf-8 -*-
import os
from collections import Counter

import numpy as np
import tensorflow as tf

import model
from embedding import BERTEmbedding, Embedding
from utils import const
from .Experiment import Experiment

__author__ = 'Jayeol Chun'


class DRSCExperiment(Experiment):
  """
  TODO (May 5): lot of redundancy
  """
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

      # effectively 768
      self.hp.word_vector_width = self.embedding.bert_config.hidden_size
      self.embedding_table = None
      self.embedding_shape = None

    else:
      # reduce number of vocab and corresponding vector entries by collecting
      # all unique tokens in dev, test and train, which slows down
      if self.hp.embedding == "random_init":
        # compiles vocab from training set
        self.processor.compile_vocab()
        vocab = self.processor.vocab
      else:
        # compiles vocab from training set AND includes vocab from all other
        # files. We do this because some external embeddings have a large vocab
        # size so words that never appear are filtered out.
        vocab = self.processor.collect_all_vocab(include_blind=True)

      tf.logging.info(f"Vocab size: {len(vocab)}")

      self.embedding = Embedding(embedding=self.hp.embedding,
                                 vocab=vocab,
                                 word_vector_width=self.hp.word_vector_width,
                                 max_arg_length=self.hp.max_arg_length)

      self.embedding_table = self.embedding.embedding_table
      self.embedding_shape = self.embedding_table.shape

      if self.hp.word_vector_width != self.hp.hidden_size:
        tf.logging.info(
          "[!] `word_vector_width` != `hidden_size`. There will be an "
          "additional linear layer that projects concatenated arg vector (of "
          "length `word_vector_width`) to `hidden_size` dimension using a "
          "learned projection weight matrix.")

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
        conn_action=self.hp.conn_action,
        embedding=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        do_pooling_first=self.hp.do_pooling_first,
        finetune_embedding=self.hp.finetune_embedding
      )
    else:
      self.model = model.Attentional(
        labels=self.labels,
        attention_type=self.hp.model,
        max_arg_length=self.hp.max_arg_length,
        word_vector_width=self.hp.word_vector_width,
        hidden_size=self.hp.hidden_size,
        num_hidden_layers=self.hp.num_hidden_layers,
        num_attention_heads=self.hp.num_attention_heads,
        learning_rate=self.hp.learning_rate,
        embedding=self.hp.embedding,
        embedding_shape=self.embedding_shape,
        optimizer=self.hp.optimizer,
        sense_type=self.hp.sense_type,
        pooling_action=self.hp.pooling_action,
        conn_action=self.hp.conn_action,
        do_pooling_first=self.hp.do_pooling_first,
        finetune_embedding=self.hp.finetune_embedding
      )

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

    init = tf.global_variables_initializer()

    tvars = tf.trainable_variables()
    tf.logging.info("***** Trainable Variables *****")
    for var in tvars:
      tf.logging.info("  name = %s, shape = %s", var.name, var.shape)

    config = tf.ConfigProto()

    # if False,pre-allocates all available GPU memory
    config.gpu_options.allow_growth = self.hp.allow_gpu_growth

    # sess handled by Experiment obj, not by each model
    self.sess = tf.Session(config=config)
    self.sess.run(init)

    train_fn(examples)

  def train_from_ids(self, examples):

    self.sess.run(
      self.model.embedding_init_op,
      feed_dict={self.model.embedding_placeholder: self.embedding_table})

    for epoch in range(self.hp.num_epochs):
      feat_batches = self.batchify(examples,
                                   batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(feat_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      for i, batch in enumerate(feat_batches):
        batch = self.embedding.convert_to_ids(batch, self.l2i)
        feed_dict = self.model.postprocess_batch_ids(batch)

        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss,
           self.model.acc], feed_dict=feed_dict)

        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

        tf.logging.info(
          "[Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
            epoch, i+1, num_batches, loss, acc))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

      if self.hp.eval_every < 0:
        # eval every end of iteration
        self.eval()

  def train_from_vals(self, examples):

    embedding_res = self.embedding.run(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    for epoch in range(self.hp.num_epochs):
      bert_batches = self.batchify(examples,
                                   batch_size=self.hp.batch_size,
                                   do_shuffle=True)
      num_batches = len(bert_batches)
      tf.logging.info(f"Created {num_batches} batches.")

      for i, batch in enumerate(bert_batches):
        feed_dict = self.model.postprocess_batch_vals(
          batch, values=bert_outputs, l2i_mapping=self.l2i,
          exid_to_feature_mapping=exid_to_feature_mapping)

        _, preds, loss, acc = self.sess.run(
          [self.model.train_op, self.model.preds, self.model.loss,
           self.model.acc], feed_dict=feed_dict)

        c = Counter(preds)
        for key in c.keys():
          tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

        tf.logging.info(
          "[TRAIN Epoch {} Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
            epoch, i+1, num_batches, loss, acc))

        if self.hp.eval_every > 0 and (i + 1) % self.hp.eval_every == 0:
          self.eval()

      if self.hp.eval_every < 0:
        # eval every end of iteration
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

  def eval_from_ids(self, examples):

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_loss = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      batch = self.embedding.convert_to_ids(batch, self.l2i)
      feed_dict = self.model.postprocess_batch_ids(batch)

      per_example_loss, loss, preds, correct, acc = \
        self.sess.run(
          [self.model.per_example_loss, self.model.loss, self.model.preds,
           self.model.correct, self.model.acc], feed_dict=feed_dict)

      # accumulate losses and prediction evaluation
      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[EVAL Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          i + 1, num_batches, loss, acc))

    tf.logging.info("[EVAL FINAL] loss: {:.3f} acc: {:.3f}".format(
      np.mean(all_loss), np.mean(all_correct)))

    # all pred outputs for dev set
    for key in all_counter.keys():
      tf.logging.info(" {:3d}: {}".format(all_counter[key], self.labels[key]))

  def eval_from_vals(self, examples):
    embedding_res = self.embedding.run(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    # collect each prediction output result, where 0: incorrect and 1: correct
    # also collect loss associated with each prediction
    # * THIS IS NECESSARY only because bert outputs for the entire batch of
    #  ~1000 relations will cause OOM issue on GPU, so move to batchified
    #  setting and handle outputs accordingly.
    all_correct = []
    all_loss = []
    all_counter = Counter()
    for i, batch in enumerate(example_batches):
      feed_dict = self.model.postprocess_batch_vals(
        batch, values=bert_outputs, l2i_mapping=self.l2i,
        exid_to_feature_mapping=exid_to_feature_mapping)

      per_example_loss, loss, preds, correct, acc = \
        self.sess.run(
          [self.model.per_example_loss, self.model.loss, self.model.preds,
           self.model.correct, self.model.acc], feed_dict=feed_dict)

      # accumulate losses and prediction evaluation
      all_loss.extend(per_example_loss)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[EVAL Batch {}/{}] loss: {:.3f} acc: {:.3f}".format(
          i + 1, num_batches, loss, acc))

    tf.logging.info("[EVAL FINAL] loss: {:.3f} acc: {:.3f}".format(
      np.mean(all_loss), np.mean(all_correct)))

    # all pred outputs for dev set
    for key in all_counter.keys():
      tf.logging.info(" {:3d}: {}".format(all_counter[key], self.labels[key]))

  ################################## PREDICT ###################################
  def predict(self):
    """See docstring for `run`

    TODO: As of May 5, no longer exporting predictions.

    Returns:
      Dict of {str of dataset_type, list of model_predictions} key-value pair
    """

    res = {}
    for dataset_type in ['test', 'blind']:
      is_test_set = dataset_type == 'test'

      predict_fn = None
      if self.hp.embedding == 'bert':
        if is_test_set:
          examples = self.processor.get_test_examples(for_bert_embedding=True)
        else:
          examples = self.processor.get_blind_examples(for_bert_embedding=True)
        predict_fn = self.predict_from_vals
      else:
        if is_test_set:
          examples = self.processor.get_test_examples()
        else:
          examples = self.processor.get_blind_examples()
        predict_fn = self.predict_from_ids

      tf.logging.info(f"Inference on {dataset_type.upper()} set")
      self.processor.remove_cache_by_key(dataset_type)

      preds = predict_fn(examples, is_test_set=is_test_set)
      res[dataset_type] = preds

    return res

  def predict_from_ids(self, examples, is_test_set=False):
    if is_test_set:
      dataset_type = "test"
    else:
      dataset_type = "blind"

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_preds = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      batch = self.embedding.convert_to_ids(batch, self.l2i)
      feed_dict = self.model.postprocess_batch_ids(batch)

      preds, correct, acc = \
        self.sess.run(
          [self.model.preds, self.model.correct, self.model.acc],
          feed_dict=feed_dict)

      all_preds.extend(preds)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[{} Batch {}/{}] acc: {:.3f}".format(
          dataset_type.upper(), i + 1, num_batches, acc))

    tf.logging.info("[{} FINAL] acc: {:.3f}".format(
      dataset_type.upper(), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(f" {self.labels[key]}: {all_counter[key]}")

    # convert label_id preds to labels
    preds_str = [self.i2l(pred) for pred in all_preds]
    return preds_str

  def predict_from_vals(self, examples, is_test_set=False):
    if is_test_set:
      dataset_type = "test"
    else:
      dataset_type = "blind"

    embedding_res = self.embedding.run(examples)
    bert_outputs, exid_to_feature_mapping = embedding_res

    example_batches = self.batchify(examples,
                                    batch_size=self.hp.batch_size,
                                    do_shuffle=False)
    num_batches = len(example_batches)
    tf.logging.info(f"Created {num_batches} batches.")

    all_correct = []
    all_preds = []
    all_counter = Counter()

    for i, batch in enumerate(example_batches):
      feed_dict = self.model.postprocess_batch_vals(
        batch, values=bert_outputs, l2i_mapping=self.l2i,
        exid_to_feature_mapping=exid_to_feature_mapping)

      preds, correct, acc = self.sess.run(
        [self.model.preds, self.model.correct, self.model.acc],
        feed_dict=feed_dict)

      all_preds.extend(preds)
      all_correct.extend(correct)

      c = Counter(preds)
      all_counter.update(c)
      for key in c.keys():
        tf.logging.info(" {:3d}: {}".format(c[key], self.labels[key]))

      tf.logging.info(
        "[{} Batch {}/{}] acc: {:.3f}".format(
          dataset_type.upper(), i + 1, num_batches,  acc))

    tf.logging.info("[{} FINAL] acc: {:.3f}".format(
      dataset_type.upper(), np.mean(all_correct)))

    for key in all_counter.keys():
      tf.logging.info(f" {self.labels[key]}: {all_counter[key]}")

    # convert preds to labels
    preds_str = [self.i2l(pred) for pred in all_preds]

    return preds_str
